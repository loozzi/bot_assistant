import datetime
from typing import TypedDict

class TaskInfo(TypedDict):
    id: int
    title: str
    description: str
    status: str
    priority: str
    created_at: datetime.date
    due_time: datetime.date | None
    finished_at: datetime.date| None
    estimate_time: datetime.timedelta | None

class TodoState(TypedDict):
    user_id: str
    user_query: str
    intent: str
    tasks: TaskInfo
    summary: str
