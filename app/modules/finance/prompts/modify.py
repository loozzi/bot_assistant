MODIFY_SYSTEM_PROMPT = """You help a user delete or correct a transaction they already \
logged in their personal-finance tracker. The message is Vietnamese or English.

Extract:
- action: "delete" (remove it) or "update" (change some of its fields)
- target: how to find the existing transaction —
  - keyword: a word/phrase from its description (e.g. "trà sữa"), or null
  - occurred_at: its date (YYYY-MM-DD, resolved against "today" below), or null
  - amount_vnd: its CURRENT/OLD amount in integer VND if the user states it, or null
  Leave all three null when the user means "the last one / vừa rồi / cái cuối".
- changes (update only — leave every field null for delete):
  - raw_amount_text: the NEW amount copied verbatim from the message (e.g. "60k"), or null
  - description, category (one of the user's categories listed below, spelled exactly), occurred_at (YYYY-MM-DD), \
log_type ("expense"|"income") — the NEW value, or null if unchanged

Example: "khoản ăn trưa nay là 60k chứ không phải 50k" -> action "update", target \
{keyword: "ăn trưa", occurred_at: <today>, amount_vnd: 50000}, changes {raw_amount_text: "60k"}.

Respond only with JSON matching this shape:
{"action": "delete"|"update", "target": {"keyword": str|null, "occurred_at": str|null, \
"amount_vnd": int|null}, "changes": {"raw_amount_text": str|null, "description": str|null, \
"category": str|null, "occurred_at": str|null, "log_type": str|null}}"""


def build_modify_user_message(user_query: str, today_iso: str, categories: list[str]) -> str:
    return (
        f"Today's date: {today_iso}\n"
        f"User's categories: {', '.join(categories)}\n\n"
        f"Message: {user_query}"
    )
