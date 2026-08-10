from types import SimpleNamespace

import pytest

from app.core.hitl import run_interruptible_subgraph


class _FakeGraph:
    """Duck-typed CompiledStateGraph stand-in: scripted aget_state snapshots
    and ainvoke results, so the checkpoint-store inconsistency (next
    populated, interrupts empty) can be reproduced without real LangGraph
    interrupt machinery."""

    def __init__(self, states, results):
        self._states = iter(states)
        self._results = iter(results)
        self.ainvoke_calls: list[dict] = []

    async def aget_state(self, config):
        return next(self._states)

    async def ainvoke(self, input, config):
        self.ainvoke_calls.append(input)
        return next(self._results)


def _snapshot(next_=(), interrupts=()):
    return SimpleNamespace(next=next_, interrupts=interrupts)


@pytest.mark.asyncio
async def test_stale_pending_checkpoint_without_interrupt_is_discarded_not_crashed():
    """A checkpoint can end up with `next` populated but no recorded
    interrupt (e.g. the graph shape changed across a redeploy while a HITL
    pause was pending, or a Redis TTL desync between the checkpoint and its
    pending-write record). This used to crash with `IndexError: tuple index
    out of range` on `snapshot.interrupts[0]`; it should instead be treated
    as unresumable and the subgraph restarted from initial_state."""
    graph = _FakeGraph(
        states=[
            _snapshot(next_=("some_node",), interrupts=()),  # stale, no answer to resume with
            _snapshot(next_=(), interrupts=()),  # fresh run reaches END
        ],
        results=[{"reply": "ok"}],
    )

    result = await run_interruptible_subgraph(graph, {"user_id": "1"}, thread_id="t1")

    assert result == {"reply": "ok"}
    assert graph.ainvoke_calls == [{"user_id": "1"}]


class _Interrupt:
    def __init__(self, value):
        self.value = value


@pytest.mark.asyncio
async def test_post_invoke_pause_is_detected_from_ainvoke_result_not_a_second_read(monkeypatch):
    """Regression for a bug the stale-checkpoint fix above introduced: the
    post-invoke pause check used to re-read state via a second aget_state()
    call. That's a separately-consistent read that can race with the write
    ainvoke() just made and report no interrupt for a pause that just
    happened, silently returning the paused (no-"reply") state instead of
    bridging it — which is exactly what produced the downstream
    `KeyError: 'reply'` in finance/agent.py. The bridge must be driven off
    ainvoke()'s own return value, and aget_state() must only be consulted
    once, before invoking."""
    bridged = []

    def _fake_interrupt(value):
        bridged.append(value)
        raise RuntimeError("bridged")

    monkeypatch.setattr("app.core.hitl.interrupt", _fake_interrupt)

    graph = _FakeGraph(
        # Only one snapshot queued: if hitl.py still called aget_state() a
        # second time after ainvoke(), that call would raise StopIteration
        # instead of the expected RuntimeError, failing this test.
        states=[_snapshot(next_=(), interrupts=())],
        results=[{"__interrupt__": [_Interrupt({"question": "confirm?"})]}],
    )

    with pytest.raises(RuntimeError, match="bridged"):
        await run_interruptible_subgraph(graph, {"user_id": "1"}, thread_id="t1")

    assert bridged == [{"question": "confirm?"}]
