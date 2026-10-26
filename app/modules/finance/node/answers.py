_AFFIRMATIVES = {"confirm", "ok", "oke", "okay", "ừ", "uh", "có", "đúng", "đồng ý", "xác nhận", "yes", "y"}


def is_affirmative(answer: object) -> bool:
    """True for the confirm button's value and the obvious typed equivalents,
    so a user who types "ok" instead of tapping isn't treated as anything else."""
    return isinstance(answer, str) and answer.strip().lower().rstrip(".!") in _AFFIRMATIVES
