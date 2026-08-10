from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import FinanceCategory

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
