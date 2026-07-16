from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest

from ..state import SearchState


def _wants_full_crawl(answer: str) -> bool:
    normalized = answer.strip().lower()
    return normalized == "yes" or "có" in normalized


async def confirm_node(state: SearchState) -> dict:
    """
    Ask the user whether to crawl full page content or just use search snippets.

    If there's nothing to confirm about (search returned no documents), skip
    the question entirely — crawling would have nothing to do either way.
    """
    documents = state.get("documents", [])
    if not documents:
        return {}

    answer = interrupt(
        HumanReviewRequest(
            question=(
                f"Tìm thấy {len(documents)} kết quả. "
                "Đọc chi tiết từng trang hay chỉ tóm tắt nhanh?"
            ),
            options=[
                {"label": "Đọc chi tiết", "value": "yes"},
                {"label": "Tóm tắt nhanh", "value": "no"},
            ],
        )
    )
    return {"skip_crawl": not _wants_full_crawl(answer)}
