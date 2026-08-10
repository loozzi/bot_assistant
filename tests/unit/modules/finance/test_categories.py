import pytest

from app.modules.finance.tools.categories import (
    DEFAULT_CATEGORIES,
    find_category_by_name,
    list_categories,
    seed_default_categories,
)
from app.modules.finance.tools.users import get_or_create_user_pk


@pytest.mark.asyncio
async def test_seed_default_categories_creates_all(db_session):
    user_pk = await get_or_create_user_pk(db_session, "1")
    await seed_default_categories(db_session, user_pk)
    await db_session.commit()

    categories = await list_categories(db_session, user_pk)
    assert len(categories) == len(DEFAULT_CATEGORIES)
    assert {c.name for c in categories} == {name for name, _group in DEFAULT_CATEGORIES}
    assert all(c.is_seeded for c in categories)


@pytest.mark.asyncio
async def test_seed_is_idempotent(db_session):
    user_pk = await get_or_create_user_pk(db_session, "2")
    await seed_default_categories(db_session, user_pk)
    await seed_default_categories(db_session, user_pk)
    await db_session.commit()

    categories = await list_categories(db_session, user_pk)
    assert len(categories) == len(DEFAULT_CATEGORIES)


@pytest.mark.asyncio
async def test_find_category_by_name_case_insensitive(db_session):
    user_pk = await get_or_create_user_pk(db_session, "3")
    await seed_default_categories(db_session, user_pk)
    await db_session.commit()

    found = await find_category_by_name(db_session, user_pk, "ĂN UỐNG")
    assert found is not None
    assert found.name == "Ăn uống"

    missing = await find_category_by_name(db_session, user_pk, "không tồn tại")
    assert missing is None
