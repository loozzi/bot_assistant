from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.db.session import get_session
from app.infra.memory.checkpointer import get_checkpointer
from app.utils.logger import get_logger

from .node.classify import classify_node
from .node.log.budget_check import log_budget_check_node
from .node.log.confirm import log_confirm_node
from .node.log.parse import log_parse_node
from .node.log.persist import log_persist_node
from .node.not_implemented import not_implemented_node
from .node.query.extract_params import query_extract_params_node
from .node.query.run_query import query_run_node
from .tools.memory import write_transaction_memory
from .tools.users import get_or_create_user_pk
from .state import FinancialState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

logger = get_logger(__name__)

def _route_after_classify(state: FinancialState) -> str:
    return state.get("sub_intent") if state.get("sub_intent") in ("log", "query") else "not_implemented"


def _route_after_confirm(state: FinancialState) -> str:
    return "persist" if state.get("confirm_answer") == "confirm" else "confirm_declined"


async def _confirm_declined_node(state: FinancialState) -> dict:
    lines = ["Được, bạn gõ lại giao dịch cho đúng nhé."]
    lines.extend(state.get("unparsed_notes", []))
    return {"reply": "\n".join(lines)}


async def _log_persist_and_followups_node(state: FinancialState) -> dict:
    """Owns the one DB session for persist -> budget_check -> memory_write."""
    try:
        async with get_session() as session:
            user_pk = await get_or_create_user_pk(session, state["user_id"])

            persist_result = await log_persist_node({**state, "session": session, "user_pk": user_pk})

            budget_result = await log_budget_check_node(
                {**state, **persist_result, "session": session, "user_pk": user_pk}
            )

            log_ids = persist_result.get("persisted_log_ids", [])
            categories = persist_result.get("persisted_categories", [])
            memory_write_args = [
                dict(
                    user_id=state["user_id"],
                    log_id=log_id,
                    category_name=category_name,
                    description=txn["description"],
                    amount_vnd=txn["amount_vnd"],
                    log_type=txn["log_type"],
                    occurred_at=txn["occurred_at"],
                )
                for log_id, txn, category_name in zip(
                    log_ids, state.get("parsed_transactions", []), categories
                )
            ]
    except Exception as exc:
        logger.warning("finance_persist_failed", error=str(exc))
        return {"reply": "Mình gặp lỗi khi lưu giao dịch, bạn thử lại sau nhé."}

    for kwargs in memory_write_args:
        await write_transaction_memory(**kwargs)

    reply_lines = [f"Đã ghi {len(log_ids)} giao dịch."]
    reply_lines.extend(budget_result.get("budget_warnings", []))
    reply_lines.extend(state.get("unparsed_notes", []))
    return {**persist_result, **budget_result, "reply": "\n".join(reply_lines)}


async def _query_run_with_session_node(state: FinancialState) -> dict:
    try:
        async with get_session() as session:
            user_pk = await get_or_create_user_pk(session, state["user_id"])
            return await query_run_node({**state, "session": session, "user_pk": user_pk})
    except Exception as exc:
        logger.warning("finance_query_failed", error=str(exc))
        return {"reply": "Mình gặp lỗi khi tra cứu, bạn thử lại sau nhé."}


def build_graph() -> StateGraph:
    builder = StateGraph(
        state_schema=FinancialState,
        name=config["agent"]["name"],
        description=config["agent"]["description"],
    )

    builder.add_node("classify", classify_node)
    builder.add_node("log_parse", log_parse_node)
    builder.add_node("log_confirm", log_confirm_node)
    builder.add_node("persist", _log_persist_and_followups_node)
    builder.add_node("confirm_declined", _confirm_declined_node)
    builder.add_node("query_extract_params", query_extract_params_node)
    builder.add_node("query_run", _query_run_with_session_node)
    builder.add_node("not_implemented", not_implemented_node)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"log": "log_parse", "query": "query_extract_params", "not_implemented": "not_implemented"},
    )

    builder.add_edge("log_parse", "log_confirm")
    builder.add_conditional_edges(
        "log_confirm", _route_after_confirm, {"persist": "persist", "confirm_declined": "confirm_declined"}
    )
    builder.add_edge("persist", END)
    builder.add_edge("confirm_declined", END)

    builder.add_edge("query_extract_params", "query_run")
    builder.add_edge("query_run", END)

    builder.add_edge("not_implemented", END)

    return builder


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(checkpointer=get_checkpointer())
    return _compiled


class FinancialAgent(BaseAgent):
    """Wraps the finance sub-intent router subgraph behind BaseAgent."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, input: AgentInput) -> AgentOutput:
        initial: FinancialState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
        }
        result = await run_interruptible_subgraph(
            get_compiled_graph(),
            initial,
            thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["reply"]}


agent = FinancialAgent()
