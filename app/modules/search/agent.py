from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.memory.checkpointer import get_checkpointer
from app.utils.logger import get_logger

from .node import confirm_node, crawl_node, search_node, summary_node
from .state import SearchState

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

builder = StateGraph(
    state_schema=SearchState,
    name=config["agent"]["name"],
    description=config["agent"]["description"],
)

builder.add_node("search", search_node)
builder.add_node("confirm", confirm_node)
builder.add_node("crawl", crawl_node)
builder.add_node("summary", summary_node)

builder.set_entry_point("search")
builder.add_edge("search", "confirm")


def _route_after_confirm(state: SearchState) -> str:
    if not state.get("documents"):
        return "summary"
    return "summary" if state.get("skip_crawl") else "crawl"


builder.add_conditional_edges("confirm", _route_after_confirm, ["crawl", "summary"])
builder.add_edge("crawl", "summary")
builder.add_edge("summary", END)

graph = builder.compile(checkpointer=get_checkpointer())


class SearchAgent(BaseAgent):
    """Wraps the search/crawl/summary subgraph behind the BaseAgent contract."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, input: AgentInput) -> AgentOutput:
        cached_reply = self._check_cache(input["message"], input.get("retrieved_memories", []))
        if cached_reply:
            logger.info("search_cache_hit", user_id=input["user_id"])
            return {"reply": cached_reply}

        initial: SearchState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
            "documents": [],
            "summary": "",
        }
        result = await run_interruptible_subgraph(
            graph,
            initial,
            thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["summary"]}

    @staticmethod
    def _check_cache(query: str, memories: list[dict]) -> str | None:
        """Return a cached answer if a near-identical query was searched recently."""
        for memory in memories:
            cached_query = memory.get("query", "")
            if cached_query and SearchAgent._similar(query, cached_query):
                return memory.get("text") or memory.get("summary_snippet")
        return None

    @staticmethod
    def _similar(query_a: str, query_b: str) -> bool:
        return query_a.strip().lower()[:20] == query_b.strip().lower()[:20]


agent = SearchAgent()
