NO_RECALL_RESULTS_MESSAGE = (
    "Chưa có gì được lưu về chủ đề này. "
    "Hãy gửi link hoặc từ khóa để tôi nghiên cứu và lưu lại."
)


def format_recall_results(results: list[dict]) -> str:
    """
    Format saved research items directly as a list, no LLM call.

    Args:
        results: List of dicts with keys: title, summary, url, timestamp, score

    Returns:
        Formatted text listing each saved item.
    """
    lines = ["Đây là những gì bạn đã từng tìm hiểu:"]
    for r in results:
        title = r.get("title") or "Untitled"
        summary = r.get("summary", "")
        url = r.get("url", "")
        timestamp = (r.get("timestamp") or "")[:10]

        lines.append("")
        header = f"• {title}"
        if timestamp:
            header += f" ({timestamp})"
        lines.append(header)
        if summary:
            lines.append(f"  {summary}")
        if url:
            lines.append(f"  {url}")

    return "\n".join(lines)
