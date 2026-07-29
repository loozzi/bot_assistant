from langgraph.types import interrupt

from app.core.hitl import HumanReviewRequest

from ..state import ResearchState


def _wants_full_crawl(answer: str) -> bool:
    normalized = answer.strip().lower()
    return normalized == "yes" or "có" in normalized


async def confirm_node(state: ResearchState) -> dict:
    """
    Ask whether to crawl full page content or just use search snippets.

    Only reached from the new_keyword flow (link_node always crawls, since
    the user gave an explicit single link with nothing to compare against).
    If there's nothing to confirm about (search returned no documents),
    skip the question entirely.
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
