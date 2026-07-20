from datetime import datetime
from zoneinfo import ZoneInfo

from predict import freeze_knockout_pick


ET = ZoneInfo("America/New_York")


def match(pick="A", status="projected"):
    return {
        "id": 73, "date": "2026-06-28", "home": "A", "away": "B",
        "pick": pick, "p_home": 0.7 if pick == "A" else 0.3,
        "p_away": 0.3 if pick == "A" else 0.7, "score": "2-0",
        "status": status,
    }


def test_knockout_pick_is_not_rewritten_after_kickoff():
    log = {}
    first = match("A")
    freeze_knockout_pick(log, first, datetime(2026, 6, 28, 10, tzinfo=ET))

    changed = match("B")
    freeze_knockout_pick(log, changed, datetime(2026, 6, 28, 14, tzinfo=ET))

    assert changed["pick"] == "A"
    assert changed["p_home"] == 0.7
    assert changed["void"] is False


def test_first_knockout_pick_seen_after_kickoff_is_void():
    log = {}
    late = match("A", "final")
    freeze_knockout_pick(log, late, datetime(2026, 6, 28, 14, tzinfo=ET))
    assert late["void"] is True
    assert log["73"]["tainted"] is True
