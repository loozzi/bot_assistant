from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import NotRequired, TypedDict

from app.utils.logger import get_logger

logger = get_logger(__name__)


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

    if snapshot.next and snapshot.interrupts:
        # Paused by a previous attempt at this same call site.
        pending = snapshot.interrupts[0].value
        answer = interrupt(pending)  # returns instantly on replay, else re-pauses the parent
        result = await graph.ainvoke(Command(resume=answer), config=config)
    else:
        if snapshot.next:
            # `next` populated with no recorded interrupt means this
            # checkpoint wasn't paused by interrupt() — e.g. the graph
            # shape changed since it was written, or its pending-write
            # record expired independently of the checkpoint (TTL
            # desync). There's no answer to resume with, so the pause is
            # unresumable; discard it and start over.
            logger.warning("hitl_stale_pending_checkpoint_discarded", thread_id=thread_id)
        result = await graph.ainvoke(initial_state, config=config)

    # Check the ainvoke() return value directly rather than re-reading state:
    # a fresh aget_state() call here is a second, separately-consistent read
    # of the checkpoint that can race with the write ainvoke() just made,
    # spuriously reporting no interrupt for a pause that just happened.
    pending_interrupts = result.get("__interrupt__")
    if pending_interrupts:
        # Just paused for the first time — bridge it up to the parent graph.
        interrupt(pending_interrupts[0].value)

    return result
