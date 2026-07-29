from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from app.core.base_agent import AgentInput
from app.core.state import AgentState
from app.infra.memory import episodic
from app.infra.memory.checkpointer import get_checkpointer
from app.infra.providers.llm_client import create_llm_client
from app.orchestrator import registry, router
from app.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_REPLY = "Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?"
PARTIAL_FAILURE_NOTE = "\n\n(Mình chưa xử lý được một phần yêu cầu của bạn.)"


def _last_human_message(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return message.content
    return ""


async def _ingest(state: AgentState) -> AgentState:
    """Fill defaults for any field the caller didn't supply."""
    return {
        "messages": state.get("messages", []),
        "user_id": state["user_id"],
        "intents": state.get("intents", []),
        "retrieved_memories": state.get("retrieved_memories", []),
        "agent_outputs": state.get("agent_outputs", {}),
        "errors": state.get("errors", []),
        "metadata": state.get("metadata", {}),
    }


async def _rehydrate_context(state: AgentState) -> AgentState:
    """Retrieve relevant past memories for the latest message, scoped to the user.

    Falls back to no context (rather than raising) on any memory-backend
    failure, so the bot keeps working when Redis/Qdrant/the embedding
    service are unavailable.
    """
    query = _last_human_message(state["messages"])
    if not query:
        return {}

    source_types = {spec["source_type"] for spec in router._load_module_specs().values()}
    try:
        memories = await episodic.retrieve_memories(state["user_id"], query, source_types)
    except Exception as exc:
        logger.warning("rehydrate_context_failed", user_id=state["user_id"], error=str(exc))
        return {}

    return {"retrieved_memories": memories}


async def _route(state: AgentState) -> AgentState:
    intents = await router.classify_intents(state)
    return {"intents": intents}


def _fan_out(state: AgentState):
    names = router.resolve_agent_names(state["intents"])
    if not names:
        return "format_response"

    message = _last_human_message(state["messages"])
    retrieved = state.get("retrieved_memories", [])
    specs = router._load_module_specs()

    sends = []
    for name in names:
        source_type = specs[name]["source_type"]
        filtered_memories = [m["payload"] for m in retrieved if m["source_type"] == source_type]
        agent_input: AgentInput = {
            "user_id": state["user_id"],
            "message": message,
            "retrieved_memories": filtered_memories,
        }
        sends.append(Send("dispatch_agent", {"_agent_name": name, "_input": agent_input}))
    return sends


async def _dispatch_agent(payload: dict) -> dict:
    agent_name = payload["_agent_name"]
    agent_input: AgentInput = payload["_input"]

    try:
        agent = registry.get(agent_name)
    except KeyError:
        logger.warning("resolved_agent_not_registered", agent=agent_name)
        return {"errors": [f"{agent_name} not available"]}

    output = await agent.run(agent_input)
    return {"agent_outputs": {agent_name: output["reply"]}}


async def _compose_reply(outputs: dict[str, str]) -> str:
    try:
        llm = create_llm_client()
        joined = "\n\n".join(f"[{name}] {reply}" for name, reply in outputs.items())
        response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Combine the following results from different assistant "
                        "modules into a single, natural reply to the user, in the "
                        "same language the results are written in. Do not mention "
                        "the module names."
                    ),
                },
                {"role": "user", "content": joined},
            ]
        )
        return response.content
    except Exception as exc:
        logger.warning("reply_composition_failed", error=str(exc))
        return "\n\n".join(outputs.values())


async def _format_response(state: AgentState) -> AgentState:
    outputs = state["agent_outputs"]

    if not outputs:
        reply = FALLBACK_REPLY
    elif len(outputs) == 1:
        reply = next(iter(outputs.values()))
    else:
        reply = await _compose_reply(outputs)

    if state.get("errors") and outputs:
        reply += PARTIAL_FAILURE_NOTE

    return {"messages": [AIMessage(content=reply)]}


async def _episodic_writer(state: AgentState) -> AgentState:
    """Write each dispatched agent's reply to Qdrant as episodic memory.

    Best-effort per intent: a write failure for one intent is logged and
    skipped rather than aborting the others or the response to the user.
    """
    outputs = state.get("agent_outputs") or {}
    if not outputs:
        return {}

    query = _last_human_message(state["messages"])
    specs = router._load_module_specs()

    for intent, reply in outputs.items():
        spec = specs.get(intent)
        if not spec:
            continue
        try:
            await episodic.write_episodic_memory(
                user_id=state["user_id"],
                query=query,
                intent=intent,
                reply=reply,
                source_type=spec["source_type"],
                default_importance=spec["default_importance"],
            )
        except Exception as exc:
            logger.warning("episodic_writer_failed", intent=intent, error=str(exc))

    return {}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("ingest", _ingest)
    graph.add_node("rehydrate_context", _rehydrate_context)
    graph.add_node("route", _route)
    graph.add_node("dispatch_agent", _dispatch_agent)
    graph.add_node("format_response", _format_response)
    graph.add_node("episodic_writer", _episodic_writer)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "rehydrate_context")
    graph.add_edge("rehydrate_context", "route")
    graph.add_conditional_edges("route", _fan_out, ["dispatch_agent", "format_response"])
    graph.add_edge("dispatch_agent", "format_response")
    graph.add_edge("format_response", "episodic_writer")
    graph.add_edge("episodic_writer", END)

    return graph


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(checkpointer=get_checkpointer())
    return _compiled
