from datetime import date
from typing import Literal

from pydantic import BaseModel


class QueryParams(BaseModel):
    metric: Literal["total", "by_category", "search"]
    period_start: date
    period_end: date
    category: str | None = None
    keyword: str | None = None
