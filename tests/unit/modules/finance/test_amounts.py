import pytest

from app.modules.finance.tools.amounts import parse_vnd


@pytest.mark.parametrize(
    "text, expected",
    [
        ("200k", 200_000),
        ("200K", 200_000),
        ("2tr", 2_000_000),
        ("2 triệu", 2_000_000),
        ("25 nghìn", 25_000),
        ("25 ngàn", 25_000),
        ("1tr2", 1_200_000),
        ("2tr5", 2_500_000),
        ("25000", 25_000),
        ("200.000", 200_000),
        ("200,000", 200_000),
        ("  50k  ", 50_000),
        ("4,1 triệu", 4_100_000),
        ("8,2 triệu", 8_200_000),
    ],
)
def test_parse_vnd_valid(text, expected):
    assert parse_vnd(text) == expected


@pytest.mark.parametrize(
    "text", ["", "abc", "$50", "50 usd", "€20", None, "2tr50", "12tr34"]
)
def test_parse_vnd_invalid(text):
    assert parse_vnd(text) is None
