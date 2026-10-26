from typing import NotRequired, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.bot import handlers
from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.orchestrator import graph as graph_module


class _AskState(TypedDict):
    user_query: str
    reply: NotRequired[str]


def _build_confirm_graph(question: str):
    async def _ask(state: _AskState) -> dict:
        answer = interrupt({"question": question})
        return {"reply": f"got:{answer}"}

    builder = StateGraph(_AskState)
    builder.add_node("ask", _ask)
    builder.set_entry_point("ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


class _PausingAgent(BaseAgent):
    """A minimal BaseAgent whose own subgraph always pauses once via
    interrupt(), bridged through the same run_interruptible_subgraph the
    real finance/search agents use."""

    def __init__(self, name: str, question: str):
        self._name = name
        self._graph = _build_confirm_graph(question)

    @property
    def name(self) -> str:
        return self._name

    async def run(self, input: AgentInput) -> AgentOutput:
        result = await run_interruptible_subgraph(
            self._graph,
            {"user_query": input["message"]},
            thread_id=f"{input['user_id']}:{self._name}",
        )
        return {"reply": result["reply"]}


@pytest.mark.asyncio
async def test_two_modules_pausing_in_one_turn_resume_sequentially(monkeypatch):
    """Regression: a single message can route to two modules that each need
    a HITL confirmation in the same turn (e.g. finance's "confirm
    transaction?" and search's "read in full or summarize?"). That leaves
    the outer graph with two pending interrupts. The old code always
    resumed with a bare Command(resume=answer); LangGraph rejects that with
    RuntimeError once more than one interrupt is pending, permanently
    wedging the thread (every future message to that user hit the same
    error). Resuming must target the interrupt currently shown to the user
    by id, and the other module's question must resurface afterward."""

    async def _fake_classify_intents(state):
        return ["finance", "search"]

    monkeypatch.setattr(graph_module.router, "classify_intents", _fake_classify_intents)

    fake_agents = {
        "finance": _PausingAgent("finance", "Xac nhan giao dich?"),
        "search": _PausingAgent("search", "Doc chi tiet hay tom tat?"),
    }
    monkeypatch.setattr(graph_module.registry, "get", lambda name: fake_agents[name])

    compiled = graph_module.build_graph().compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(handlers, "get_compiled_graph", lambda: compiled)

    user_id = "u-multi"
    sent: list[tuple[str, object]] = []

    async def _send(text: str, reply_markup=None) -> None:
        sent.append((text, reply_markup))

    # Drive through _reply_or_prompt (the real display path) rather than
    # reading result["__interrupt__"] directly, so the test exercises the
    # actual guarantee the fix relies on: display and resume must agree on
    # which interrupt is "first".
    first = await handlers._run_turn(user_id, new_text="mua ca phe 35k va tim hieu ve X")
    await handlers._reply_or_prompt(_send, user_id, first)
    assert len(sent) == 1
    first_question = sent[0][0]
    first_module = "finance" if first_question.startswith("Xac nhan") else "search"
    second_module = "search" if first_module == "finance" else "finance"

    # Old behavior: this line used to raise
    # `RuntimeError: When there are multiple pending interrupts, you must
    # specify the interrupt id when resuming.`
    second = await handlers._run_turn(user_id, resume_answer="yes")
    await handlers._reply_or_prompt(_send, user_id, second)
    assert len(sent) == 2  # the other module's question resurfaced

    third = await handlers._run_turn(user_id, resume_answer="no")
    assert not third.get("__interrupt__")
    assert third["agent_outputs"] == {first_module: "got:yes", second_module: "got:no"}
