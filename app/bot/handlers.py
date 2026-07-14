from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from langchain_core.messages import HumanMessage

from app.bot.keyboards import main_menu_keyboard
from app.core.state import AgentState
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
        "<b>Phân tích:</b> 'Phân tích chi tiêu tháng này'\n\n"
        "Dùng /menu để xem menu chính.",
        parse_mode="HTML",
    )


@router.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    await message.answer("Chọn chức năng:", reply_markup=main_menu_keyboard())


# ---------------------------------------------------------------------------
# Main message handler — delegates to orchestrator
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_text(message: Message, user_id: str) -> None:
    assert message.text is not None

    thinking = await message.answer("⏳ Đang xử lý...")

    try:
        state: AgentState = {
            "messages": [HumanMessage(content=message.text)],
            "user_id": user_id,
        }

        graph = get_compiled_graph()
        result: AgentState = await graph.ainvoke(state)

        last = next(
            (m for m in reversed(result["messages"]) if getattr(m, "type", None) == "ai"),
            None,
        )
        reply = last.content if last else "Mình chưa có câu trả lời cho điều này."

    except Exception as exc:
        logger.exception("orchestrator_error", user_id=user_id, error=str(exc))
        reply = "❌ Có lỗi xảy ra. Bạn thử lại sau nhé!"

    await thinking.delete()
    await message.answer(reply)


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
    }
    await callback.message.answer(prompts.get(module, "Bạn cần gì?"))  # type: ignore[union-attr]
    await callback.answer()
