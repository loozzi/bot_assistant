import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, create_model

from app.core.state import AgentState
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

_MODULES_PATH = Path(__file__).resolve().parent.parent / "modules"


@functools.lru_cache(maxsize=1)
def _load_module_specs() -> dict[str, dict]:
    """Read every app/modules/<name>/config.yaml into {name: {description, examples, keywords}}."""
    specs: dict[str, dict] = {}
    for config_path in sorted(_MODULES_PATH.glob("*/config.yaml")):
        cfg = yaml.safe_load(config_path.read_text())
        agent_cfg = cfg.get("agent", {})
        if not agent_cfg.get("enabled", True):
            continue
        name = agent_cfg.get("name", config_path.parent.name)
        routing_cfg = cfg.get("routing", {})
        memory_cfg = cfg.get("memory", {})
        specs[name] = {
            "description": agent_cfg.get("description", "").strip(),
            "examples": routing_cfg.get("examples", []),
            "keywords": routing_cfg.get("keywords", {}),
            "source_type": memory_cfg.get("source_type", name),
            "default_importance": memory_cfg.get("default_importance", 0.5),
        }
    return specs


def _build_system_prompt(specs: dict[str, dict]) -> str:
    sections = []
    for name, spec in specs.items():
        lines = [f"### {name}", spec["description"]]
        if spec["examples"]:
            lines.append("Examples:")
            lines.extend(f"- {ex}" for ex in spec["examples"])
        keywords = [*spec["keywords"].get("vi", []), *spec["keywords"].get("en", [])]
        if keywords:
            lines.append(f"Keywords: {', '.join(keywords)}")
        sections.append("\n".join(lines))

    modules_block = "\n\n".join(sections)
    return (
        "You are an intent classifier for a Telegram personal-assistant bot. "
        "Given the user's latest message, choose every module that applies — "
        "most messages need only one, but choose more than one when the "
        'message clearly asks for multiple distinct things. If no module '
        'clearly applies, choose "unknown".\n\n'
        f"Available modules:\n\n{modules_block}\n\n"
        'Respond only with a JSON object of the form {"intents": ["<module_name>", ...]}.'
    )


def _build_intent_schema(specs: dict[str, dict]) -> type[BaseModel]:
    labels = (*specs.keys(), "unknown")
    return create_model("IntentClassification", intents=(list[Literal[labels]], ...))


async def classify_intents(state: AgentState) -> list[str]:
    """Return one or more intent labels for the latest user message, via LLM classification."""
    last = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    if last is None:
        return ["unknown"]

    specs = _load_module_specs()
    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(_build_intent_schema(specs), method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": _build_system_prompt(specs)},
                {"role": "user", "content": getattr(last, "content", "")},
            ]
        )
        return result.intents or ["unknown"]
    except Exception as exc:
        logger.warning("intent_classification_failed", error=str(exc))
        return ["unknown"]


def resolve_agent_names(intents: list[str]) -> list[str]:
    specs = _load_module_specs()
    seen: list[str] = []
    for intent in intents:
        if intent != "unknown" and intent in specs and intent not in seen:
            seen.append(intent)
    return seen
