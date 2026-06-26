from abc import ABC, abstractmethod

from app.core.state import AgentState


class BaseAgent(ABC):
    """All feature agents must subclass this and implement `run`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier used for routing and logging."""
        ...

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """Process the current state and return an updated state."""
        ...
