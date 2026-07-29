from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.memory.checkpointer import get_checkpointer

from .node import (
    classify_node,
    confirm_node,
    crawl_node,
    link_node,
    recall_node,
    search_node,
    summarize_and_save_node,
)
from .state import ResearchState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

builder = StateGraph(
    state_schema=ResearchState,
    name=config["agent"]["name"],
    description=config["agent"]["description"],
)

builder.add_node("classify", classify_node)
builder.add_node("link", link_node)
builder.add_node("search", search_node)
builder.add_node("confirm", confirm_node)
builder.add_node("crawl", crawl_node)
builder.add_node("summarize_and_save", summarize_and_save_node)
builder.add_node("recall", recall_node)

builder.set_entry_point("classify")


def _route_after_classify(state: ResearchState) -> str:
    mode = state.get("mode", "new_keyword")
    if mode == "new_link":
        return "link"
    if mode == "recall":
        return "recall"
    return "search"


builder.add_conditional_edges("classify", _route_after_classify, ["link", "search", "recall"])
builder.add_edge("link", "crawl")
builder.add_edge("search", "confirm")


def _route_after_confirm(state: ResearchState) -> str:
    if not state.get("documents"):
        return "summarize_and_save"
    return "summarize_and_save" if state.get("skip_crawl") else "crawl"


builder.add_conditional_edges("confirm", _route_after_confirm, ["crawl", "summarize_and_save"])
builder.add_edge("crawl", "summarize_and_save")
builder.add_edge("summarize_and_save", END)
builder.add_edge("recall", END)

graph = builder.compile(checkpointer=get_checkpointer())


class ResearchAgent(BaseAgent):
    """Wraps the classify/link/search/confirm/crawl/summarize/recall subgraph behind BaseAgent."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, input: AgentInput) -> AgentOutput:
        initial: ResearchState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
            "documents": [],
            "reply": "",
        }
        result = await run_interruptible_subgraph(
            graph,
            initial,
            thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["reply"]}


agent = ResearchAgent()
