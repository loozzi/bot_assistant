from datetime import date

from app.modules.finance.tools.dates import _ICT, today_ict


def test_today_ict_returns_a_date():
    assert isinstance(today_ict(), date)


def test_uses_asia_ho_chi_minh_zone():
    """Finding 6: the shared 'today' helper must be pinned to Asia/Ho_Chi_Minh
    (ICT, UTC+7), not UTC or server-local time."""
    assert _ICT.key == "Asia/Ho_Chi_Minh"
