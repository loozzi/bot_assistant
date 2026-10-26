from typing import Literal

from pydantic import BaseModel, Field, model_validator

SubIntent = Literal[
    "log", "query", "modify", "category", "budget", "goal", "split", "advice", "fallback"
]


class ClassifyResult(BaseModel):
    sub_intents: list[SubIntent] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_single(cls, data):
        # Tolerate the single-label shape ({"sub_intent": "log"}) — an LLM
        # may still answer that way, and it's the natural way to build one.
        if isinstance(data, dict) and "sub_intents" not in data and "sub_intent" in data:
            return {"sub_intents": [data["sub_intent"]]}
        return data
