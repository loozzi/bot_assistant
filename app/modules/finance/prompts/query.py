QUERY_SYSTEM_PROMPT = """You extract structured query parameters from a question about \
past spending/income. Resolve relative periods (e.g. "tháng này", "this month", "hôm nay") \
against the "today" date given below into explicit period_start/period_end ISO dates.

metric must be one of:
- "total": how much was spent/earned in the period (optionally filtered by category)
- "by_category": breakdown of spending by category in the period
- "search": look up transactions matching a keyword (e.g. "how much did I spend on trà sữa")

Set category to the Vietnamese category name if the user names one, else null.
Set keyword to the search term if metric is "search", else null.

Respond only with JSON matching:
{"metric": "total"|"by_category"|"search", "period_start": "YYYY-MM-DD", \
"period_end": "YYYY-MM-DD", "category": str|null, "keyword": str|null}"""


def build_query_user_message(user_query: str, today_iso: str) -> str:
    return f"Today's date: {today_iso}\n\nQuestion: {user_query}"
