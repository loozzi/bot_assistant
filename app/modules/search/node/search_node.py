from ..state import SearchState


async def search_node(state: SearchState) -> dict:
    """
    Perform a search based on the user's query and return relevant documents.
    """
    return {
        "documents": [
            {
                "title": "[stub] Placeholder result",
                "url": "https://example.invalid/stub",
                "raw_text": (
                    "This is placeholder content — search agent is not yet "
                    "wired to a real search provider."
                ),
            }
        ]
    }