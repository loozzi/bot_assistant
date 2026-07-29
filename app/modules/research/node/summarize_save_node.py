from app.infra.providers.embedding import embed_text
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ..prompts.summary import NO_RESULTS_MESSAGE, SUMMARY_SYSTEM_PROMPT, format_fallback_summary
from ..schema.research_summary import ResearchSummary
from ..state import ResearchState
from ..tools import get_default_importance, upsert_research_point

logger = get_logger(__name__)


async def summarize_and_save_node(state: ResearchState) -> dict:
    """
    Summarize the documents, then best-effort save the result to Qdrant.

    If no documents are available, returns NO_RESULTS_MESSAGE without any
    LLM call. If the LLM structured-output call fails, falls back to a
    plain title/URL list. If saving to Qdrant fails for any reason, logs a
    warning and still returns the summary — the user's answer must not be
    lost just because persistence failed.
    """
    documents = state.get("documents", [])
    user_query = state.get("user_query", "")

    if not documents:
        return {"reply": NO_RESULTS_MESSAGE}

    doc_content_lines = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "")
        url = doc.get("url", "")
        raw_text = doc.get("raw_text", "")
        truncated = raw_text[:2000] if raw_text else ""

        doc_content_lines.append(f"Document {i}:")
        if title:
            doc_content_lines.append(f"Title: {title}")
        if url:
            doc_content_lines.append(f"URL: {url}")
        if truncated:
            doc_content_lines.append(f"Content: {truncated}")
        doc_content_lines.append("")

    doc_block = "\n".join(doc_content_lines)
    user_message = f"Query: {user_query}\n\n{doc_block}"

    try:
        llm = create_llm_client()
        classifier = llm.with_structured_output(ResearchSummary, method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
        reply = result.render()
        summary_text = result.answer
        topics = result.topics
    except Exception as exc:
        logger.warning("summarize_save_node_llm_call_failed", error=str(exc))
        reply = format_fallback_summary(documents)
        summary_text = reply
        topics = []

    title = documents[0].get("title") or user_query
    url = documents[0].get("url", "")

    try:
        vector = await embed_text(summary_text, prefix="passage: ")
        await upsert_research_point(
            user_id=state["user_id"],
            url=url,
            title=title,
            summary=summary_text,
            topics=topics,
            importance=get_default_importance(),
            vector=vector,
        )
    except Exception as exc:
        logger.warning("research_save_failed", error=str(exc))

    return {"reply": reply}
