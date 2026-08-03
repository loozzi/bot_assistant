from app.infra.providers.llm_client import create_llm_client
from app.utils.logger import get_logger

from ...prompts.query import QUERY_SYSTEM_PROMPT, build_query_user_message
from ...schema.query_params import QueryParams
from ...tools.dates import today_ict

logger = get_logger(__name__)


def _today_iso() -> str:
    return today_ict().isoformat()


async def query_extract_params_node(state: dict) -> dict:
    try:
        llm = create_llm_client()
        extractor = llm.with_structured_output(QueryParams, method="json_mode")
        result: QueryParams = await extractor.ainvoke(
            [
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_query_user_message(state["user_query"], _today_iso()),
                },
            ]
        )
        return {"query_params": result.model_dump(mode="json")}
    except Exception as exc:
        logger.warning("finance_query_extract_failed", error=str(exc))
        return {"query_params": None}
