from ...tools.categories import (
    CategoryError,
    create_category,
    delete_category,
    list_categories,
    merge_categories,
    rename_category,
    seed_default_categories,
)

_GROUP_LABELS = {
    "essential": "Thiết yếu",
    "lifestyle": "Phong cách sống",
    "savings_debt": "Tiết kiệm & nợ",
    "other": "Khác",
}


async def _render_list(session, user_pk: int) -> str:
    await seed_default_categories(session, user_pk)
    by_group: dict[str, list[str]] = {}
    for category in await list_categories(session, user_pk):
        by_group.setdefault(category.group, []).append(category.name)
    lines = ["Danh mục của bạn:"]
    for group, label in _GROUP_LABELS.items():
        if by_group.get(group):
            lines.append(f"• {label}: {', '.join(sorted(by_group[group]))}")
    return "\n".join(lines)


async def category_apply_node(state: dict) -> dict:
    """Requires state["session"] and state["user_pk"] (injected by agent.py)."""
    session = state["session"]
    user_pk = state["user_pk"]
    params = state["category_params"]
    action = params["action"]
    name = (params.get("name") or "").strip()
    target = (params.get("target_name") or "").strip()

    if action == "list":
        return {"reply": await _render_list(session, user_pk)}

    try:
        if action == "create":
            category = await create_category(session, user_pk, name, params.get("group") or "other")
            return {"reply": f"Đã thêm danh mục '{category.name}'."}
        if action == "rename":
            await rename_category(session, user_pk, name, target)
            return {"reply": f"Đã đổi tên '{name}' thành '{target}'."}
        if action == "merge":
            moved = await merge_categories(session, user_pk, name, target)
            return {"reply": f"Đã gộp '{name}' vào '{target}' ({moved} giao dịch được chuyển)."}
        moved = await delete_category(session, user_pk, name)
        return {"reply": f"Đã xoá danh mục '{name}' ({moved} giao dịch chuyển sang 'Khác')."}
    except CategoryError as exc:
        return {"reply": str(exc)}
