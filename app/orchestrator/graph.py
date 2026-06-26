from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.core.state import AgentState
from app.orchestrator import registry, router
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def _route(state: AgentState) -> AgentState:
    intent = await router.classify_intent(state)
    return {**state, "intent": intent}


async def _dispatch(state: AgentState) -> AgentState:
    agent_name = router.resolve_agent_name(state["intent"])

    if agent_name is None:
        logger.warning("unresolved_intent", intent=state["intent"])
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn không?")
        return {**state, "messages": state["messages"] + [fallback]}

    agent = registry.get(agent_name)
    return await agent.run(state)


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("route", _route)
    graph.add_node("dispatch", _dispatch)

    graph.set_entry_point("route")
    graph.add_edge("route", "dispatch")
    graph.add_edge("dispatch", END)

    return graph


# Compiled graph singleton — call build_graph().compile() at startup
_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled
