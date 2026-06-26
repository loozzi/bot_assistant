from app.core.base_agent import BaseAgent
from app.utils.logger import get_logger

logger = get_logger(__name__)

_registry: dict[str, BaseAgent] = {}


def register(agent: BaseAgent) -> None:
    if agent.name in _registry:
        raise ValueError(f"Agent '{agent.name}' is already registered")
    _registry[agent.name] = agent
    logger.info("agent_registered", agent=agent.name)


def get(name: str) -> BaseAgent:
    if name not in _registry:
        raise KeyError(f"No agent registered with name '{name}'")
    return _registry[name]


def all_agents() -> dict[str, BaseAgent]:
    return dict(_registry)
