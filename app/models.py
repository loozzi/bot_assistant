from datetime import date, datetime
from typing import Literal

from sqlalchemy import BigInteger, CheckConstraint, Date, ForeignKey, Index, UniqueConstraint, func, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_JSON_BLOB = postgresql.JSONB().with_variant(sqlite.JSON(), "sqlite")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FinanceCategory(Base):
    __tablename__ = "finance_categories"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_finance_categories_user_name"),
        CheckConstraint(
            "\"group\" IN ('essential', 'lifestyle', 'savings_debt', 'other')",
            name="ck_finance_categories_group",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column(default="")
    group: Mapped[str] = mapped_column(default="other")
    is_seeded: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FinanceLog(Base):
    __tablename__ = "finance_logs"
    __table_args__ = (
        CheckConstraint("log_type IN ('income', 'expense')", name="ck_finance_logs_log_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    amount: Mapped[int] = mapped_column(BigInteger)
    description: Mapped[str] = mapped_column()
    message: Mapped[str] = mapped_column()
    log_type: Mapped[Literal["income", "expense"]] = mapped_column()

    occurred_at: Mapped[date] = mapped_column(Date, index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    extra: Mapped[dict] = mapped_column(_JSON_BLOB, default=dict)

    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("finance_categories.id", ondelete="SET NULL"), default=None, nullable=True
    )


class FinanceBudget(Base):
    __tablename__ = "finance_budgets"
    __table_args__ = (
        UniqueConstraint("user_id", "category_id", name="uq_finance_budgets_user_category"),
        Index(
            "uq_finance_budgets_user_overall",
            "user_id",
            unique=True,
            postgresql_where=text("category_id IS NULL"),
            sqlite_where=text("category_id IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("finance_categories.id", ondelete="CASCADE"), nullable=True
    )
    amount: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class FinanceGoal(Base):
    __tablename__ = "finance_goals"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'achieved', 'archived')", name="ck_finance_goals_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    description: Mapped[str] = mapped_column()
    target_amount: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    deadline: Mapped[date | None] = mapped_column(Date, nullable=True)
    monthly_target: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[Literal["active", "achieved", "archived"]] = mapped_column(default="active")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class FinanceDebt(Base):
    __tablename__ = "finance_debts"
    __table_args__ = (
        CheckConstraint("direction IN ('owed_to_me', 'i_owe')", name="ck_finance_debts_direction"),
        CheckConstraint("status IN ('open', 'settled')", name="ck_finance_debts_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    counterparty: Mapped[str] = mapped_column()
    direction: Mapped[Literal["owed_to_me", "i_owe"]] = mapped_column()
    amount_total: Mapped[int] = mapped_column(BigInteger)
    amount_outstanding: Mapped[int] = mapped_column(BigInteger)
    description: Mapped[str] = mapped_column(default="")
    status: Mapped[Literal["open", "settled"]] = mapped_column(default="open")
    extra: Mapped[dict] = mapped_column(_JSON_BLOB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    settled_at: Mapped[datetime | None] = mapped_column(nullable=True)
