from typing import Literal

from pydantic import BaseModel


class ModeClassification(BaseModel):
    """Structured output for classify_node: which research flow applies."""
    mode: Literal["new_link", "new_keyword", "recall"]
