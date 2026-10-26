import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.orchestrator import graph as graph_module


class _FakeAgent:
    def __init__(self, reply: str):
        self._reply = reply

    async def run(self, agent_input):
        return {"reply": self._reply}


@pytest.mark.asyncio
async def test_agent_outputs_do_not_leak_across_turns(monkeypatch):
    """Regression: `agent_outputs`/`errors` use accumulating reducers
    (`operator.or_`/`operator.add`) so parallel dispatch_agent branches can
    fan back in within one turn — but the graph's thread persists for the
    user's whole conversation. Without an explicit per-turn reset, a module
    dispatched in an earlier turn kept reappearing in `agent_outputs`
    forever and got re-merged into replies for turns that never routed to
    it (e.g. a stale "search" reply about an old topic bleeding into a
    later, unrelated "finance" turn)."""
    turn = {"n": 1}

    async def _fake_classify_intents(state):
        return ["search"] if turn["n"] == 1 else ["finance"]

    monkeypatch.setattr(graph_module.router, "classify_intents", _fake_classify_intents)

    fake_agents = {"search": _FakeAgent("search reply"), "finance": _FakeAgent("finance reply")}
    monkeypatch.setattr(graph_module.registry, "get", lambda name: fake_agents[name])

    compiled = graph_module.build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "u1"}}

    first = await compiled.ainvoke(
        {"messages": [HumanMessage(content="tìm kiếm gì đó")], "user_id": "u1"}, config=config
    )
    assert first["agent_outputs"] == {"search": "search reply"}

    turn["n"] = 2
    second = await compiled.ainvoke(
        {"messages": [HumanMessage(content="mua cafe 35k")], "user_id": "u1"}, config=config
    )

    assert second["intents"] == ["finance"]
    assert second["agent_outputs"] == {"finance": "finance reply"}
