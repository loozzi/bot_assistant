import importlib
import pkgutil
from pathlib import Path

from app.core.base_agent import BaseAgent
from app.utils.logger import get_logger

logger = get_logger(__name__)

_registry: dict[str, BaseAgent] = {}
_MODULES_PATH = Path(__file__).resolve().parent.parent / "modules"


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


def discover_and_register() -> None:
    """Import every app.modules.<name>.agent module and register its `agent`.

    Modules without an agent.py are skipped (expected for modules that
    haven't been implemented yet, e.g. journal/finance/insight/todo today).
    A module's config.yaml `agent.enabled: false` prevents registration
    without requiring the module to be deleted.
    """
    for _, module_name, is_pkg in pkgutil.iter_modules([str(_MODULES_PATH)]):
        if not is_pkg:
            continue

        agent_module_path = _MODULES_PATH / module_name / "agent.py"
        if not agent_module_path.exists():
            logger.debug("module_has_no_agent", module=module_name)
            continue

        module = importlib.import_module(f"app.modules.{module_name}.agent")
        agent = getattr(module, "agent", None)
        if not isinstance(agent, BaseAgent):
            logger.warning("module_agent_attribute_missing_or_invalid", module=module_name)
            continue

        module_config = getattr(module, "config", {})
        enabled = module_config.get("agent", {}).get("enabled", True)
        if not enabled:
            logger.info("module_agent_disabled", module=module_name)
            continue

        register(agent)
