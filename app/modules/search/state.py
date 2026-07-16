from typing import TypedDict

from typing_extensions import NotRequired


class DocumentInfo(TypedDict):
    title: str
    url: str
    raw_text: str


class SearchState(TypedDict):
    user_id: str
    user_query: str
    documents: list[DocumentInfo]
    summary: str
    skip_crawl: NotRequired[bool]
