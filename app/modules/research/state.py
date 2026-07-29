from typing import Literal

from typing_extensions import NotRequired, TypedDict


class DocumentInfo(TypedDict):
    title: str
    url: str
    raw_text: str


class ResearchState(TypedDict):
    user_id: str
    user_query: str
    mode: NotRequired[Literal["new_link", "new_keyword", "recall"]]
    url: NotRequired[str]
    documents: list[DocumentInfo]
    skip_crawl: NotRequired[bool]
    reply: str
