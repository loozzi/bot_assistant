from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ModifyTarget(BaseModel):
    """Which saved transaction the user means. All empty = the latest one."""

    keyword: str | None = None
    occurred_at: date | None = None
    amount_vnd: int | None = None


class ModifyChanges(BaseModel):
    """New values for an update — only the fields the user changed are set."""

    raw_amount_text: str | None = None
    description: str | None = None
    category: str | None = None
    occurred_at: date | None = None
    log_type: Literal["income", "expense"] | None = None


class ModifyParams(BaseModel):
    action: Literal["delete", "update"]
    target: ModifyTarget = Field(default_factory=ModifyTarget)
    changes: ModifyChanges = Field(default_factory=ModifyChanges)
