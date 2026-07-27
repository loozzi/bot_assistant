NO_RESULTS_MESSAGE = (
    "Không tìm thấy nội dung để tóm tắt. "
    "Vui lòng thử lại với link hoặc từ khóa khác."
)

SUMMARY_SYSTEM_PROMPT = """You are a research assistant that summarizes web content for later recall.

Your task:
1. Read the provided documents
2. Answer the user's query based ONLY on the content in these documents
3. If the documents don't contain enough information to answer confidently, say so
4. Always respond in the same language as the user's query
5. Be concise and factual

Format your response as JSON with these fields:
- answer: A direct, concise answer to the user's question
- key_points: A list of 2-4 important supporting points (can be empty if not relevant)
- topics: A list of 2-5 short topic tags describing this content, for future retrieval"""


def format_fallback_summary(documents: list[dict]) -> str:
    """
    Create a fallback summary when the LLM call fails.

    Shows document titles and URLs without LLM processing.

    Args:
        documents: List of document dicts with keys: title, url, raw_text

    Returns:
        Formatted text with title + URL list for top 3 documents
    """
    if not documents:
        return NO_RESULTS_MESSAGE

    lines = ["Nội dung đã tìm được:"]
    for doc in documents[:3]:
        title = doc.get("title", "Untitled")
        url = doc.get("url", "")
        if url:
            lines.append(f"• [{title}]({url})")
        else:
            lines.append(f"• {title}")

    return "\n".join(lines)
