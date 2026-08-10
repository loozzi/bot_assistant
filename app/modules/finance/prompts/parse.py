PARSE_SYSTEM_PROMPT = """You extract structured transactions from a Vietnamese/English \
message describing spending or income. A single message may describe multiple transactions.

For each transaction, produce:
- amount_vnd: the amount in integer VND (e.g. "200k" -> 200000, "2 triệu" -> 2000000)
- raw_amount_text: the exact substring of the original message that states the amount \
  (e.g. "200k", "2 triệu", "25000") — copy it verbatim, do not normalize it yourself
- description: a short description of what was bought/received
- log_type: "expense" or "income"
- suggested_category: your best-guess category name in Vietnamese (e.g. "Ăn uống", "Đi lại")
- occurred_at: an ISO date (YYYY-MM-DD), resolved against the "today" date given below \
  (e.g. "hôm qua" = yesterday, no date mentioned = today)

If any amount is stated in a non-VND currency (e.g. "$50", "50 usd", "20 eur"), do NOT \
produce a transaction for it — instead add a short note about it to unparsed_notes.

If the message describes no transactions at all, return an empty transactions list.

Respond only with JSON matching this shape:
{"transactions": [{"amount_vnd": int, "raw_amount_text": str, "description": str, \
"log_type": "expense"|"income", "suggested_category": str, "occurred_at": "YYYY-MM-DD"}], \
"unparsed_notes": [str]}"""


def build_parse_user_message(user_query: str, today_iso: str) -> str:
    return f"Today's date: {today_iso}\n\nMessage: {user_query}"
