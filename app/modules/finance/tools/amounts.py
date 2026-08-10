import re

_FOREIGN_MARKERS = ("$", "usd", "eur", "€", "gbp", "£")

_MILLION_WITH_TENTHS = re.compile(r"^(\d+)\s*tr\s*(\d)$")
_MILLION = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(triệu|tr)$")
_THOUSAND = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|k)$")
_PLAIN_NUMBER = re.compile(r"^[\d.,]+$")


def parse_vnd(text: str | None) -> int | None:
    """Deterministically parse a colloquial Vietnamese amount into integer VND.

    Returns None for empty/unparseable input or anything carrying a
    non-VND currency marker — callers treat None as "could not verify".
    """
    if not text:
        return None

    normalized = text.strip().lower()
    if not normalized:
        return None
    if any(marker in normalized for marker in _FOREIGN_MARKERS):
        return None

    match = _MILLION_WITH_TENTHS.match(normalized)
    if match:
        whole, tenths = match.groups()
        return int(whole) * 1_000_000 + int(tenths) * 100_000

    match = _MILLION.match(normalized)
    if match:
        value = float(match.group(1).replace(",", "."))
        return round(value * 1_000_000)

    match = _THOUSAND.match(normalized)
    if match:
        value = float(match.group(1).replace(",", "."))
        return round(value * 1_000)

    if _PLAIN_NUMBER.match(normalized):
        digits = re.sub(r"[.,]", "", normalized)
        if digits.isdigit() and digits:
            return int(digits)

    return None
