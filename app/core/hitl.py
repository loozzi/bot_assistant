from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import NotRequired, TypedDict


class HumanReviewOption(TypedDict):
    label: str
    value: str


class HumanReviewRequest(TypedDict):
    question: str
    options: NotRequired[list[HumanReviewOption]]


async def run_interruptible_subgraph(
    graph: CompiledStateGraph,
    initial_state: dict[str, Any],
    *,
    thread_id: str,
) -> dict[str, Any]:
    """Run a checkpointed subgraph, bridging any pause up to the parent graph.

    Must be awaited from inside a node of a checkpointed parent graph — the
    bridging `interrupt()` call below relies on that parent's checkpoint
    context. Supports exactly one pending pause per invocation; a subgraph
    that needs to pause a second time within the same run isn't supported yet.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if snapshot.next:
        # Paused by a previous attempt at this same call site.
        pending = snapshot.interrupts[0].value
        answer = interrupt(pending)  # returns instantly on replay, else re-pauses the parent
        result = await graph.ainvoke(Command(resume=answer), config=config)
    else:
        result = await graph.ainvoke(initial_state, config=config)

    snapshot = await graph.aget_state(config)
    if snapshot.next:
        # Just paused for the first time — bridge it up to the parent graph.
        interrupt(snapshot.interrupts[0].value)

    return result
