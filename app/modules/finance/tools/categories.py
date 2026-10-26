from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceBudget, FinanceCategory, FinanceLog

DEFAULT_CATEGORIES: list[tuple[str, str]] = [
    ("Ăn uống", "essential"),
    ("Đi lại", "essential"),
    ("Nhà cửa", "essential"),
    ("Hóa đơn & tiện ích", "essential"),
    ("Sức khỏe", "essential"),
    ("Giáo dục", "essential"),
    ("Mua sắm", "lifestyle"),
    ("Giải trí", "lifestyle"),
    ("Tiết kiệm & đầu tư", "savings_debt"),
    ("Khác", "other"),
]


def category_names(state: dict) -> list[str]:
    """The user's category names as loaded into graph state by agent.py's
    load_categories node — the defaults if that load didn't happen/failed."""
    return state.get("available_categories") or [name for name, _ in DEFAULT_CATEGORIES]


async def list_categories(session: AsyncSession, user_pk: int) -> list[FinanceCategory]:
    result = await session.execute(
        select(FinanceCategory).where(FinanceCategory.user_id == user_pk)
    )
    return list(result.scalars().all())


async def seed_default_categories(session: AsyncSession, user_pk: int) -> None:
    """Create the default category set for a user, unless they already have any."""
    existing = await list_categories(session, user_pk)
    if existing:
        return

    for name, group in DEFAULT_CATEGORIES:
        session.add(
            FinanceCategory(
                user_id=user_pk, name=name, description="", group=group, is_seeded=True
            )
        )
    await session.flush()


async def find_category_by_name(
    session: AsyncSession, user_pk: int, name: str
) -> FinanceCategory | None:
    result = await session.execute(
        select(FinanceCategory).where(FinanceCategory.user_id == user_pk)
    )
    search_name = name.strip().lower()
    for category in result.scalars():
        if category.name.lower() == search_name:
            return category
    return None


_FALLBACK_CATEGORY = "Khác"


class CategoryError(Exception):
    """A category operation the user asked for can't be done; str() is a
    user-facing Vietnamese explanation."""


async def _require(session: AsyncSession, user_pk: int, name: str) -> FinanceCategory:
    category = await find_category_by_name(session, user_pk, name)
    if category is None:
        raise CategoryError(f"Không có danh mục '{name}'.")
    return category


def _protect_fallback(category: FinanceCategory) -> None:
    if category.name.lower() == _FALLBACK_CATEGORY.lower():
        raise CategoryError(f"Danh mục '{_FALLBACK_CATEGORY}' là mặc định, không đổi/xoá được.")


async def create_category(
    session: AsyncSession, user_pk: int, name: str, group: str = "other"
) -> FinanceCategory:
    await seed_default_categories(session, user_pk)
    name = name.strip()
    if not name:
        raise CategoryError("Tên danh mục không được để trống.")
    if await find_category_by_name(session, user_pk, name) is not None:
        raise CategoryError(f"Danh mục '{name}' đã có rồi.")
    category = FinanceCategory(user_id=user_pk, name=name, description="", group=group)
    session.add(category)
    await session.flush()
    return category


async def rename_category(session: AsyncSession, user_pk: int, old_name: str, new_name: str) -> FinanceCategory:
    await seed_default_categories(session, user_pk)
    category = await _require(session, user_pk, old_name)
    _protect_fallback(category)
    new_name = new_name.strip()
    existing = await find_category_by_name(session, user_pk, new_name)
    if existing is not None and existing.id != category.id:
        raise CategoryError(f"Danh mục '{new_name}' đã có rồi — muốn gộp thì nói \"gộp {old_name} vào {new_name}\" nhé.")
    category.name = new_name
    await session.flush()
    return category


async def merge_categories(session: AsyncSession, user_pk: int, source_name: str, target_name: str) -> int:
    """Move every log of `source` into `target`, then delete `source`.

    The source's budget moves to the target only if the target has none —
    otherwise the target's own budget wins and the source's is dropped.
    Returns how many logs were moved."""
    await seed_default_categories(session, user_pk)
    source = await _require(session, user_pk, source_name)
    target = await _require(session, user_pk, target_name)
    _protect_fallback(source)
    if source.id == target.id:
        raise CategoryError("Hai danh mục này là một.")

    moved = (
        await session.execute(
            update(FinanceLog)
            .where(FinanceLog.user_id == user_pk, FinanceLog.category_id == source.id)
            .values(category_id=target.id)
        )
    ).rowcount

    source_budget = (
        await session.execute(select(FinanceBudget).where(FinanceBudget.category_id == source.id))
    ).scalar_one_or_none()
    if source_budget is not None:
        target_has_budget = (
            await session.execute(select(FinanceBudget.id).where(FinanceBudget.category_id == target.id))
        ).first() is not None
        if target_has_budget:
            await session.delete(source_budget)
        else:
            source_budget.category_id = target.id

    await session.delete(source)
    await session.flush()
    return moved


async def delete_category(session: AsyncSession, user_pk: int, name: str) -> int:
    """Deleting = merging into "Khác", so no log ever loses its category."""
    return await merge_categories(session, user_pk, name, _FALLBACK_CATEGORY)
