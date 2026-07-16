from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.core.hitl import HumanReviewOption


def cancel_keyboard(label: str = "Huỷ") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=label, callback_data="cancel")]]
    )


def confirm_keyboard(
    confirm_label: str = "Xác nhận",
    cancel_label: str = "Huỷ",
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text=confirm_label, callback_data="confirm")
    builder.button(text=cancel_label, callback_data="cancel")
    builder.adjust(2)
    return builder.as_markup()


def main_menu_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="📓 Nhật ký", callback_data="menu:journal")
    builder.button(text="💰 Chi tiêu", callback_data="menu:finance")
    builder.button(text="🔍 Tìm kiếm", callback_data="menu:search")
    builder.button(text="📊 Phân tích", callback_data="menu:insight")
    builder.adjust(2)
    return builder.as_markup()


def hitl_keyboard(options: list[HumanReviewOption]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for option in options:
        builder.button(text=option["label"], callback_data=f"hitl:{option['value']}")
    builder.button(text="Huỷ", callback_data="hitl_cancel")
    builder.adjust(2)
    return builder.as_markup()


def pagination_keyboard(
    page: int,
    total_pages: int,
    prefix: str,
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    if page > 0:
        builder.button(text="◀ Trước", callback_data=f"{prefix}:page:{page - 1}")
    if page < total_pages - 1:
        builder.button(text="Sau ▶", callback_data=f"{prefix}:page:{page + 1}")
    builder.adjust(2)
    return builder.as_markup()
