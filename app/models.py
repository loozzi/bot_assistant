from datetime import datetime

from sqlalchemy import BigInteger, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from typing import Literal

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FinanceCategory(Base):
    __tablename__ = "finance_categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column()
    name: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class FinanceLog(Base):
    __tablename__ = "finance_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column()

    amount: Mapped[float] = mapped_column()
    description: Mapped[str] = mapped_column()
    message: Mapped[str] = mapped_column()
    log_type: Mapped[Literal['income', 'expense']] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    metadata: Mapped[dict] = mapped_column(default={})

    category_id: Mapped[int] = mapped_column(default=None, nullable=True)  
