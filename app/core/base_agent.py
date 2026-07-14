from abc import ABC, abstractmethod
from typing import TypedDict


class AgentInput(TypedDict):
    user_id: str
    message: str
    retrieved_memories: list[dict]


class AgentOutput(TypedDict):
    reply: str


class BaseAgent(ABC):
    """All feature agents must subclass this and implement `run`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier used for routing and logging."""
        ...

    @abstractmethod
    async def run(self, input: AgentInput) -> AgentOutput:
        """Process minimal input and return the agent's own reply only."""
        ...
