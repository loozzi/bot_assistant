from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.memory.checkpointer import get_checkpointer

from .node import confirm_node, crawl_node, search_node, summary_node
from .state import SearchState

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


agent = SearchAgent()
