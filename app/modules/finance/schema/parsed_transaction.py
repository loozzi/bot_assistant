from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ParsedTransaction(BaseModel):
    amount_vnd: int
    raw_amount_text: str
    description: str
    log_type: Literal["income", "expense"]
    suggested_category: str
    occurred_at: date

    @field_validator("amount_vnd")
    @classmethod
    def amount_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("amount_vnd must be positive")
        return value


class ParseResult(BaseModel):
    transactions: list[ParsedTransaction] = Field(default_factory=list)
    unparsed_notes: list[str] = Field(default_factory=list)
