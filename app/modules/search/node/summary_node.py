from ..state import SearchState
from ..schema import SearchSummary
from ..prompts.summary import SUMMARY_SYSTEM_PROMPT, NO_RESULTS_MESSAGE, format_fallback_summary
from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def summary_node(state: SearchState) -> dict:
    """
    Summarize the documents into a structured answer with key points and sources.
    
    If no documents are available, returns NO_RESULTS_MESSAGE without LLM call.
    If LLM structured-output call fails, falls back to plain title/URL list.
    """
    documents = state.get("documents", [])
    user_query = state.get("user_query", "")
    
    # No documents: return fallback immediately
    if not documents:
        return {"summary": NO_RESULTS_MESSAGE}
    
    # Build user message from query + document content
    doc_content_lines = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "")
        url = doc.get("url", "")
        raw_text = doc.get("raw_text", "")
        
        # Truncate content to ~2000 chars to bound prompt size
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
        classifier = llm.with_structured_output(SearchSummary, method="json_mode")
        result = await classifier.ainvoke(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
        return {"summary": result.render()}
    except Exception as exc:
        logger.warning("summary_node_llm_call_failed", error=str(exc))
        # Fall back to plain list of titles and URLs
        fallback = format_fallback_summary(documents)
        return {"summary": fallback}