import asyncio

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from langchain_core.messages import HumanMessage
from langgraph.types import Command as ResumeCommand

from app.bot.keyboards import hitl_keyboard, main_menu_keyboard
from app.core.state import AgentState
from app.infra.memory.checkpointer import get_checkpointer
from app.orchestrator import registry
from app.orchestrator.graph import get_compiled_graph
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = Router(name="main")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message, user_id: str) -> None:
    await message.answer(
        "👋 Xin chào! Mình là trợ lý cá nhân của bạn.\n\n"
        "Bạn có thể:\n"
        "• 📓 Ghi nhật ký & cảm xúc\n"
        "• 💰 Theo dõi chi tiêu\n"
        "• 🔍 Tìm kiếm thông tin\n"
        "• 🔬 Lưu & tra cứu lại nghiên cứu (link, từ khoá)\n"
        "• 📊 Xem phân tích xu hướng\n\n"
        "Hoặc chỉ cần nhắn tin tự nhiên — mình sẽ hiểu!",
        reply_markup=main_menu_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "📖 <b>Hướng dẫn sử dụng</b>\n\n"
        "<b>Nhật ký:</b> Kể về ngày của bạn, cảm xúc, suy nghĩ\n"
        "<b>Chi tiêu:</b> 'Mua cà phê 45k', 'Tốn 200k tiền ăn'\n"
        "<b>Tìm kiếm:</b> 'Tìm ...', 'Search ...'\n"
        "<b>Nghiên cứu:</b> 'Lưu link này lại ...', 'Tôi từng đọc gì về ... chưa?'\n"
        "<b>Phân tích:</b> 'Phân tích chi tiêu tháng này'\n\n"
        "Dùng /menu để xem menu chính. Dùng /cancel để huỷ câu hỏi đang chờ.",
        parse_mode="HTML",
    )


@router.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    await message.answer("Chọn chức năng:", reply_markup=main_menu_keyboard())


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, user_id: str) -> None:
    async with _get_user_lock(user_id):
        await _clear_pending_review(user_id)
    await message.answer("Đã huỷ yêu cầu trước đó. Bạn cần gì tiếp theo?")


# ---------------------------------------------------------------------------
# Shared turn-running helpers
# ---------------------------------------------------------------------------

# Same user_id's graph turns share one Redis checkpoint thread — running two
# concurrently (e.g. the user sends a second message before the first reply
# arrives) races on that checkpoint's reads/writes and corrupts it. One lock
# per user_id serializes everything that touches the graph/checkpoint for
# that user; the bot stays fully concurrent across *different* users.
_user_locks: dict[str, asyncio.Lock] = {}


def _get_user_lock(user_id: str) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock


async def _clear_pending_review(user_id: str) -> None:
    checkpointer = get_checkpointer()
    await checkpointer.adelete_thread(user_id)
    for agent_name in registry.all_agents():
        await checkpointer.adelete_thread(f"{user_id}:{agent_name}")


def _first_pending_interrupt(snapshot):
    """The first interrupt belonging to a task that hasn't resolved yet.

    A checkpoint's tasks each carry every interrupt they've ever raised
    (snapshot.interrupts / task.interrupts), and LangGraph doesn't drop an
    already-answered one until *every* task in that superstep resolves —
    so once one of several parallel-paused modules has been answered but
    another hasn't, both still show up as "interrupted". task.result is
    None exactly while a task is still actually blocked, so it's the only
    reliable way to tell which interrupt is still live.
    """
    for task in snapshot.tasks:
        if task.result is None and task.interrupts:
            return task.interrupts[0]
    return None


async def _run_turn(
    user_id: str,
    *,
    new_text: str | None = None,
    resume_answer: str | None = None,
) -> AgentState:
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": user_id}}

    if resume_answer is not None:
        snapshot = await graph.aget_state(config)
        pending = _first_pending_interrupt(snapshot)
        if pending is not None:
            # More than one module can pause for confirmation within the same
            # turn (e.g. finance's "confirm transaction?" and search's "read
            # in full or summarize?" both firing for one message). LangGraph
            # then requires resuming a specific interrupt by id rather than a
            # bare value — this is the same one _reply_or_prompt last showed
            # the user (nothing else runs against this thread between showing
            # it and this resume — _get_user_lock serializes that). Any other
            # still-pending interrupt is untouched and resurfaces once this
            # one is answered.
            return await graph.ainvoke(
                ResumeCommand(resume={pending.id: resume_answer}), config=config
            )
        # Nothing actually pending (e.g. a stale button tap after the flow
        # already finished) — there's no answer to resume, so treat the
        # incoming text as a fresh message instead of crashing.
        new_text = new_text or resume_answer

    state: AgentState = {
        "messages": [HumanMessage(content=new_text)],
        "user_id": user_id,
    }
    return await graph.ainvoke(state, config=config)


async def _reply_or_prompt(send, user_id: str, result: AgentState) -> None:
    # Re-derive "is there a pending interrupt, and which one is first" from
    # aget_state() rather than trusting result["__interrupt__"] directly, so
    # this agrees with the resume logic in _run_turn above about which
    # interrupt is still live (see _first_pending_interrupt's docstring).
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": user_id}}
    snapshot = await graph.aget_state(config)
    pending = _first_pending_interrupt(snapshot)

    if pending is not None:
        payload = pending.value
        options = payload.get("options")
        keyboard = hitl_keyboard(options) if options else None
        await send(payload["question"], reply_markup=keyboard)
        return

    last = next(
        (m for m in reversed(result["messages"]) if getattr(m, "type", None) == "ai"),
        None,
    )
    reply = last.content if last else "Mình chưa có câu trả lời cho điều này."
    await send(reply)


# ---------------------------------------------------------------------------
# Main message handler — delegates to orchestrator
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_text(message: Message, user_id: str) -> None:
    assert message.text is not None

    thinking = await message.answer("⏳ Đang xử lý...")

    try:
        async with _get_user_lock(user_id):
            graph = get_compiled_graph()
            config = {"configurable": {"thread_id": user_id}}
            snapshot = await graph.aget_state(config)

            if snapshot.next:
                result = await _run_turn(user_id, resume_answer=message.text)
            else:
                result = await _run_turn(user_id, new_text=message.text)

        await thinking.delete()
        await _reply_or_prompt(message.answer, user_id, result)

    except Exception as exc:
        logger.exception("orchestrator_error", user_id=user_id, error=str(exc))
        await thinking.delete()
        await message.answer("❌ Có lỗi xảy ra. Bạn thử lại sau nhé!")


# ---------------------------------------------------------------------------
# Callback queries
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "cancel")
async def cb_cancel(callback: CallbackQuery) -> None:
    await callback.message.delete()  # type: ignore[union-attr]
    await callback.answer("Đã huỷ.")


@router.callback_query(F.data == "confirm")
async def cb_confirm(callback: CallbackQuery) -> None:
    await callback.answer("Đã xác nhận.")


@router.callback_query(F.data.startswith("menu:"))
async def cb_menu(callback: CallbackQuery, user_id: str) -> None:
    module = callback.data.split(":")[1]  # type: ignore[union-attr]
    prompts = {
        "journal": "Hãy kể về ngày hôm nay của bạn.",
        "finance": "Bạn muốn ghi chi tiêu gì?",
        "search": "Bạn muốn tìm kiếm gì?",
        "insight": "Bạn muốn phân tích gì?",
        "research": "Gửi link hoặc từ khoá bạn muốn nghiên cứu và lưu lại.",
    }
    await callback.message.answer(prompts.get(module, "Bạn cần gì?"))  # type: ignore[union-attr]
    await callback.answer()


@router.callback_query(F.data == "hitl_cancel")
async def cb_hitl_cancel(callback: CallbackQuery, user_id: str) -> None:
    async with _get_user_lock(user_id):
        await _clear_pending_review(user_id)
    await callback.message.edit_text("Đã huỷ yêu cầu trước đó.")  # type: ignore[union-attr]
    await callback.answer()


@router.callback_query(F.data.startswith("hitl:"))
async def cb_hitl(callback: CallbackQuery, user_id: str) -> None:
    value = callback.data.split(":", 1)[1]  # type: ignore[union-attr]
    await callback.answer()

    try:
        async with _get_user_lock(user_id):
            result = await _run_turn(user_id, resume_answer=value)
        await callback.message.delete()  # type: ignore[union-attr]
        await _reply_or_prompt(callback.message.answer, user_id, result)  # type: ignore[union-attr]
    except Exception as exc:
        logger.exception("orchestrator_error", user_id=user_id, error=str(exc))
        await callback.message.answer("❌ Có lỗi xảy ra. Bạn thử lại sau nhé!")  # type: ignore[union-attr]
