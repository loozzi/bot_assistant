import operator
from typing import Annotated, Any, TypedDict
from typing_extensions import NotRequired

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    intents: list[str]
    retrieved_memories: NotRequired[list[dict]]
    agent_outputs: Annotated[dict[str, Any], operator.or_]
    errors: Annotated[list[str], operator.add]
    metadata: NotRequired[dict[str, Any]]
