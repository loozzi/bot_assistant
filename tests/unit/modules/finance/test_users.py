import pytest
from sqlalchemy import select

from app.models import User
from app.modules.finance.tools.users import get_or_create_user_pk


@pytest.mark.asyncio
async def test_creates_user_on_first_call(db_session):
    pk = await get_or_create_user_pk(db_session, "555")
    assert isinstance(pk, int)

    row = (await db_session.execute(select(User).where(User.id == pk))).scalar_one()
    assert row.telegram_id == 555


@pytest.mark.asyncio
async def test_returns_same_pk_on_repeat_call(db_session):
    first = await get_or_create_user_pk(db_session, "777")
    second = await get_or_create_user_pk(db_session, "777")
    assert first == second

    count = (await db_session.execute(select(User))).all()
    assert len(count) == 1
