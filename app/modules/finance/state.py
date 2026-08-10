from typing_extensions import NotRequired, TypedDict



class FinancialState(TypedDict):
    """State for the FinancialAgent's internal subgraph."""

    user_id: str
    user_query: str

    sub_intent: NotRequired[str]

    # log branch
    parsed_transactions: NotRequired[list[dict]]
    unparsed_notes: NotRequired[list[str]]
    confirm_answer: NotRequired[str]
    persisted_log_ids: NotRequired[list[int]]
    persisted_categories: NotRequired[list[str]]
    budget_warnings: NotRequired[list[str]]

    # query branch
    query_params: NotRequired[dict | None]

    reply: NotRequired[str]
