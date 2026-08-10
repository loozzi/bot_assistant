from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User


async def get_or_create_user_pk(session: AsyncSession, telegram_id: str) -> int:
    """Resolve a Telegram id (string, as carried through AgentInput) to users.id."""
    tg_id = int(telegram_id)
    existing = (
        await session.execute(select(User).where(User.telegram_id == tg_id))
    ).scalar_one_or_none()
    if existing:
        return existing.id

    user = User(telegram_id=tg_id)
    session.add(user)
    await session.flush()
    return user.id
