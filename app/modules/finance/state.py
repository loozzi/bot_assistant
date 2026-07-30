from typing_extensions import TypedDict

class FinancialState(TypedDict):
    """State for the FinancialAgent."""

    user_id: int
    user_query: str