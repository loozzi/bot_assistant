from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.infra.db.session import get_session
from app.utils.logger import get_logger

from .node.category.apply import category_apply_node
from .node.category.confirm import category_confirm_node, needs_confirmation
from .node.category.extract_params import category_extract_params_node
from .node.classify import classify_node
from .node.log.budget_check import log_budget_check_node
from .node.log.confirm import log_confirm_node
from .node.log.edit import MAX_EDIT_ROUNDS, log_apply_edit_node, log_ask_edit_node
from .node.log.parse import log_parse_node
from .node.log.persist import log_persist_node
from .node.modify.extract_params import modify_extract_params_node
from .node.modify.find import modify_find_node
from .node.modify.review import apply_changes, modify_confirm_node, modify_select_node, render_log
from .node.not_implemented import not_implemented_node
from .node.query.extract_params import query_extract_params_node
from .node.query.run_query import query_run_node
from .tools.categories import list_categories, seed_default_categories
from .tools.memory import delete_transaction_memory, write_transaction_memory
from .tools.transactions import category_name_of, delete_log, get_log, update_log
from .tools.users import get_or_create_user_pk
from .state import FinancialState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

logger = get_logger(__name__)

_ENTRY_NODES = {
    "log": "log_parse",
    "modify": "modify_extract_params",
    "category": "category_extract_params",
}


def _route_after_classify(state: FinancialState) -> str:
    sub_intent = state.get("sub_intent")
    if sub_intent in _ENTRY_NODES:
        return "load_categories"
    return "query" if sub_intent == "query" else "not_implemented"


async def _next_intent_node(state: FinancialState) -> dict:
    """Every branch ends here. Collect its reply, then either start the next
    queued sub-intent or join all replies into the final one."""
    replies = [*state.get("replies", []), *([state["reply"]] if state.get("reply") else [])]
    pending = state.get("pending_intents", [])
    if pending:
        return {"replies": replies, "reply": None, "sub_intent": pending[0], "pending_intents": pending[1:]}
    return {"replies": replies, "reply": "\n\n".join(replies), "sub_intent": None}


def _route_after_next_intent(state: FinancialState) -> str:
    return "__end__" if state.get("sub_intent") is None else _route_after_classify(state)


def _route_after_load_categories(state: FinancialState) -> str:
    return _ENTRY_NODES[state["sub_intent"]]


async def _load_categories_node(state: FinancialState) -> dict:
    """The user's category names, for the LLM prompts that pick a category.
    On a DB error the prompts fall back to the default set (see
    tools/categories.py:category_names) rather than failing the turn."""
    try:
        async with get_session() as session:
            user_pk = await get_or_create_user_pk(session, state["user_id"])
            await seed_default_categories(session, user_pk)
            names = [category.name for category in await list_categories(session, user_pk)]
    except Exception as exc:
        logger.warning("finance_load_categories_failed", error=str(exc))
        return {}
    return {"available_categories": names}


def _route_after_category_extract(state: FinancialState) -> str:
    if state.get("reply"):
        return "__end__"
    return "category_confirm" if needs_confirmation(state["category_params"]) else "category_apply"


async def _category_apply_with_session_node(state: FinancialState) -> dict:
    try:
        async with get_session() as session:
            user_pk = await get_or_create_user_pk(session, state["user_id"])
            return await category_apply_node({**state, "session": session, "user_pk": user_pk})
    except Exception as exc:
        logger.warning("finance_category_apply_failed", error=str(exc))
        return {"reply": "Mình gặp lỗi khi cập nhật danh mục, bạn thử lại sau nhé."}


def _route_after_confirm(state: FinancialState) -> str:
    answer = state.get("confirm_answer")
    if answer == "confirm":
        return "persist"
    if not state.get("parsed_transactions") or state.get("edit_rounds", 0) >= MAX_EDIT_ROUNDS:
        return "confirm_declined"
    # "edit" is the button; anything else is a correction typed straight in.
    return "ask_edit" if answer == "edit" else "apply_edit"


def _route_after_apply_edit(state: FinancialState) -> str:
    return "log_confirm" if state.get("parsed_transactions") else "__end__"


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


def _route_modify_after_find(state: FinancialState) -> str:
    if state.get("reply"):
        return "__end__"
    return "modify_select" if len(state["modify_candidates"]) > 1 else "modify_confirm"


def _end_if_replied(next_node: str):
    def _route(state: FinancialState) -> str:
        return "__end__" if state.get("reply") else next_node

    return _route


async def _modify_find_with_session_node(state: FinancialState) -> dict:
    try:
        async with get_session() as session:
            user_pk = await get_or_create_user_pk(session, state["user_id"])
            return await modify_find_node({**state, "session": session, "user_pk": user_pk})
    except Exception as exc:
        logger.warning("finance_modify_find_failed", error=str(exc))
        return {"reply": "Mình gặp lỗi khi tìm giao dịch, bạn thử lại sau nhé."}


async def _modify_apply_node(state: FinancialState) -> dict:
    """Delete or update the chosen log, then bring the Qdrant memory in line.

    The log is re-fetched by id scoped to this user (it may have been
    removed between the confirm prompt and the answer)."""
    chosen = state["modify_candidates"][0]
    action = state["modify_action"]
    changes = state.get("modify_changes", {})
    try:
        async with get_session() as session:
            user_pk = await get_or_create_user_pk(session, state["user_id"])
            log = await get_log(session, user_pk, chosen["id"])
            if log is None:
                return {"reply": "Giao dịch này không còn nữa, có thể đã bị xoá trước đó."}
            if action == "delete":
                await delete_log(session, log)
            else:
                await update_log(session, user_pk, log, changes)
                memory_kwargs = dict(
                    user_id=state["user_id"],
                    log_id=log.id,
                    category_name=await category_name_of(session, log),
                    description=log.description,
                    amount_vnd=log.amount,
                    log_type=log.log_type,
                    occurred_at=log.occurred_at.isoformat(),
                )
    except Exception as exc:
        logger.warning("finance_modify_apply_failed", action=action, error=str(exc))
        return {"reply": "Mình gặp lỗi khi cập nhật giao dịch, bạn thử lại sau nhé."}

    await delete_transaction_memory(state["user_id"], chosen["id"])
    if action == "delete":
        return {"reply": f"Đã xoá: {render_log(chosen)}"}

    await write_transaction_memory(**memory_kwargs)
    updated = {**apply_changes(chosen, changes), "category": memory_kwargs["category_name"]}
    return {"reply": f"Đã sửa: {render_log(updated)}"}


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
    builder.add_node("load_categories", _load_categories_node)
    builder.add_node("log_parse", log_parse_node)
    builder.add_node("log_confirm", log_confirm_node)
    builder.add_node("persist", _log_persist_and_followups_node)
    builder.add_node("ask_edit", log_ask_edit_node)
    builder.add_node("apply_edit", log_apply_edit_node)
    builder.add_node("confirm_declined", _confirm_declined_node)
    builder.add_node("query_extract_params", query_extract_params_node)
    builder.add_node("query_run", _query_run_with_session_node)
    builder.add_node("modify_extract_params", modify_extract_params_node)
    builder.add_node("modify_find", _modify_find_with_session_node)
    builder.add_node("modify_select", modify_select_node)
    builder.add_node("modify_confirm", modify_confirm_node)
    builder.add_node("modify_apply", _modify_apply_node)
    builder.add_node("category_extract_params", category_extract_params_node)
    builder.add_node("category_confirm", category_confirm_node)
    builder.add_node("category_apply", _category_apply_with_session_node)
    builder.add_node("not_implemented", not_implemented_node)
    builder.add_node("next_intent", _next_intent_node)

    builder.set_entry_point("classify")
    intent_entries = {
        "load_categories": "load_categories",
        "query": "query_extract_params",
        "not_implemented": "not_implemented",
    }
    builder.add_conditional_edges("classify", _route_after_classify, intent_entries)
    builder.add_conditional_edges(
        "load_categories", _route_after_load_categories, {node: node for node in _ENTRY_NODES.values()}
    )

    builder.add_edge("log_parse", "log_confirm")
    builder.add_conditional_edges(
        "log_confirm",
        _route_after_confirm,
        {
            "persist": "persist",
            "ask_edit": "ask_edit",
            "apply_edit": "apply_edit",
            "confirm_declined": "confirm_declined",
        },
    )
    builder.add_edge("ask_edit", "apply_edit")
    builder.add_conditional_edges(
        "apply_edit", _route_after_apply_edit, {"log_confirm": "log_confirm", "__end__": "next_intent"}
    )
    builder.add_edge("persist", "next_intent")
    builder.add_edge("confirm_declined", "next_intent")

    builder.add_edge("query_extract_params", "query_run")
    builder.add_edge("query_run", "next_intent")

    builder.add_conditional_edges(
        "modify_extract_params",
        _end_if_replied("modify_find"),
        {"modify_find": "modify_find", "__end__": "next_intent"},
    )
    builder.add_conditional_edges(
        "modify_find",
        _route_modify_after_find,
        {"modify_select": "modify_select", "modify_confirm": "modify_confirm", "__end__": "next_intent"},
    )
    builder.add_conditional_edges(
        "modify_select",
        _end_if_replied("modify_confirm"),
        {"modify_confirm": "modify_confirm", "__end__": "next_intent"},
    )
    builder.add_conditional_edges(
        "modify_confirm",
        _end_if_replied("modify_apply"),
        {"modify_apply": "modify_apply", "__end__": "next_intent"},
    )
    builder.add_edge("modify_apply", "next_intent")

    builder.add_conditional_edges(
        "category_extract_params",
        _route_after_category_extract,
        {"category_confirm": "category_confirm", "category_apply": "category_apply", "__end__": "next_intent"},
    )
    builder.add_conditional_edges(
        "category_confirm",
        _end_if_replied("category_apply"),
        {"category_apply": "category_apply", "__end__": "next_intent"},
    )
    builder.add_edge("category_apply", "next_intent")

    builder.add_edge("not_implemented", "next_intent")
    builder.add_conditional_edges(
        "next_intent", _route_after_next_intent, {**intent_entries, "__end__": END}
    )

    return builder


_compiled = None


def get_compiled_graph():
    """Compiled WITHOUT its own checkpointer on purpose: invoked from inside
    the orchestrator's dispatch node, it runs as a native LangGraph subgraph
    and checkpoints into the parent's thread. That lets the log flow pause
    any number of times per run (confirm -> edit -> confirm ...), each pause
    surfacing in the parent snapshot with its own interrupt id — which
    app/core/hitl.py's single-pause bridge can't do."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
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
        # Must be awaited from inside a checkpointed parent graph node — a
        # pause raises GraphInterrupt up through this call and the parent
        # resumes us (replaying back into the paused node) on the next turn.
        result = await get_compiled_graph().ainvoke(initial)
        return {"reply": result["reply"]}


agent = FinancialAgent()
