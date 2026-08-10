from typing import Literal

from pydantic import BaseModel

SubIntent = Literal["log", "query", "budget", "goal", "split", "advice", "fallback"]


class ClassifyResult(BaseModel):
    sub_intent: SubIntent
