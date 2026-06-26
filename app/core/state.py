from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    intent: str                     # which agent was invoked
    retrieved_memories: list[dict]  # context retrieved from Qdrant
    metadata: dict[str, Any]        # auxiliary information
