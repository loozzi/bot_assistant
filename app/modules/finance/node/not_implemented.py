_MESSAGES = {
    "budget": "Tính năng đặt ngân sách sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "goal": "Tính năng mục tiêu tài chính sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "split": "Tính năng chia tiền sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "advice": "Tính năng tư vấn chi tiêu sẽ sớm có mặt! Hiện tại mình chưa hỗ trợ được.",
    "fallback": "Mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn về khoản thu/chi hoặc câu hỏi chi tiêu không?",
}


async def not_implemented_node(state: dict) -> dict:
    reply = _MESSAGES.get(state.get("sub_intent", "fallback"), _MESSAGES["fallback"])
    return {"reply": reply}
