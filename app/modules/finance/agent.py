from pathlib import Path

import yaml
from langgraph.graph import END, StateGraph

from app.core.base_agent import AgentInput, AgentOutput, BaseAgent
from app.core.hitl import run_interruptible_subgraph
from app.infra.memory.checkpointer import get_checkpointer

from .node import placeholder_node
from .state import FinancialState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(_CONFIG_PATH.read_text())

builder = StateGraph(
    state_schema=FinancialState,
    name=config["agent"]["name"],
    description=config["agent"]["description"],
)

builder.add_node("placeholder", placeholder_node)
builder.set_entry_point("placeholder")
builder.add_edge("placeholder", END)

graph = builder.compile(checkpointer=get_checkpointer())


class FinancialAgent(BaseAgent):
    """Wraps the classify/link/search/confirm/crawl/summarize/recall subgraph behind BaseAgent."""

    @property
    def name(self) -> str:
        return config["agent"]["name"]

    async def run(self, input: AgentInput) -> AgentOutput:
        initial: FinancialState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
            "reply": "",
        }
        result = await run_interruptible_subgraph(
            graph,
            initial,
            thread_id=f"{input['user_id']}:{self.name}",
        )
        return {"reply": result["reply"]}


agent = FinancialAgent()
