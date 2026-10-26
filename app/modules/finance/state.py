from typing_extensions import NotRequired, TypedDict



class FinancialState(TypedDict):
    """State for the FinancialAgent's internal subgraph."""

    user_id: str
    user_query: str

    sub_intent: NotRequired[str | None]
    # further sub-intents from the same message, run after the current one
    pending_intents: NotRequired[list[str]]
    # replies of the sub-intents already finished this run
    replies: NotRequired[list[str]]
    # user's category names, loaded once for the branches whose LLM calls need them
    available_categories: NotRequired[list[str]]

    # log branch
    parsed_transactions: NotRequired[list[dict]]
    unparsed_notes: NotRequired[list[str]]
    confirm_answer: NotRequired[str]
    edit_instruction: NotRequired[str]
    edit_rounds: NotRequired[int]
    edit_notice: NotRequired[list[str]]
    persisted_log_ids: NotRequired[list[int]]
    persisted_categories: NotRequired[list[str]]
    budget_warnings: NotRequired[list[str]]

    # modify branch (delete/update an already-saved log)
    modify_action: NotRequired[str]
    modify_target: NotRequired[dict]
    modify_changes: NotRequired[dict]
    modify_candidates: NotRequired[list[dict]]
    modify_confirmed: NotRequired[bool]

    # category branch
    category_params: NotRequired[dict]

    # query branch
    query_params: NotRequired[dict | None]

    reply: NotRequired[str | None]
