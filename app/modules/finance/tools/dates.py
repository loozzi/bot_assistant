from datetime import date, datetime
from zoneinfo import ZoneInfo

_ICT = ZoneInfo("Asia/Ho_Chi_Minh")


def today_ict() -> date:
    return datetime.now(_ICT).date()
