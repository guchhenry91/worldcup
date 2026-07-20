"""Player picks: squad-thinness guard, freezing, and grading.

The regressions these lock down all really happened:
  - a one-man squad absorbing its team's whole lambda and publishing as a 72.8%
    anytime scorer (the highest number on the board, and entirely an artefact),
  - grading a penalty goal as a miss because the rates feed counts NON-penalty
    goals, when an anytime-scorer pick plainly wins on a penalty,
  - an own goal counting as both a shot and a goal for the man who put it in.
"""
import pandas as pd
import pytest

from leagues import picks, players, props
from leagues.publish import _player_pick_publishable


# ---------------------------------------------------------------- thin squads
def _rates(counts):
    rows = [{"team": t, "player": f"{t}{i}", "pos": "FW", "nineties": 10.0,
             "rate90": 0.5, "shots90": 2.0, "sot_ratio": 0.35}
            for t, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows)


def test_thin_squad_is_flagged_but_a_full_one_is_not():
    rates = _rates({"Schalke 04": 1, "Bayern Munich": 11})
    assert props.thin_squads(rates, ["Schalke 04", "Bayern Munich"], 6) == ["Schalke 04"]


def test_team_with_no_data_is_not_called_thin():
    """Zero players is a DIFFERENT failure, handled by missing_squads. Conflating
    them would double-report the same club and, worse, imply an empty card is the
    dangerous case -- it is the visible one."""
    rates = _rates({"Bayern Munich": 11})
    assert props.thin_squads(rates, ["Bayern Munich", "Elversberg"], 6) == []


def test_one_man_squad_would_absorb_the_whole_team_lambda():
    """The bug itself, demonstrated: without the guard a single player is handed
    every goal his team is expected to score."""
    rates = _rates({"Solo": 1, "Other": 11})
    out = props.match_props(rates, "Solo", "Other", 1.3, 1.0)
    solo = [p for p in out if p["team"] == "Solo"]
    assert len(solo) == 1
    assert solo[0]["lambda_goals"] == pytest.approx(1.3)
    assert solo[0]["anytime_pct"] > 70          # absurd, hence MIN_SQUAD_FOR_PROPS


# ------------------------------------------------------------------- freezing
def test_lock_prop_writes_once_and_never_rewrites():
    log = {}
    ko = pd.Timestamp("2026-08-22T14:00:00Z")
    first = picks.lock_prop(log, "k", market="goal", player="Kane", team="Bayern",
                            p_pick=0.48, confidence=2, kickoff=ko,
                            now=ko - pd.Timedelta(hours=3))
    again = picks.lock_prop(log, "k", market="goal", player="Kane", team="Bayern",
                            p_pick=0.99, confidence=5, kickoff=ko,
                            now=ko - pd.Timedelta(hours=1))
    assert again == first and first["p_pick"] == 0.48   # the later 0.99 is ignored
    assert first["tainted"] is False


def test_a_pick_first_locked_after_kickoff_is_tainted():
    ko = pd.Timestamp("2026-08-22T14:00:00Z")
    e = picks.lock_prop({}, "k", market="goal", player="X", team="T", p_pick=0.5,
                        confidence=3, kickoff=ko, now=ko + pd.Timedelta(hours=4))
    assert e["tainted"] is True
    assert picks.grade_prop(e, {"goals": 2})["graded"] == "void"   # never counted


# -------------------------------------------------------------------- grading
@pytest.mark.parametrize("market,actual,expected", [
    ("goal",  {"goals": 1}, "correct"),
    ("goal",  {"goals": 0}, "wrong"),
    ("shots", {"shots": 2}, "correct"),
    ("shots", {"shots": 1}, "wrong"),     # the line is 2+, not 1+
    ("sot",   {"sot": 1},   "correct"),
    ("sot",   {"sot": 0},   "wrong"),
])
def test_each_market_grades_on_its_own_line(market, actual, expected):
    e = {"market": market, "player": "X", "team": "T", "tainted": False}
    assert picks.grade_prop(e, actual)["graded"] == expected


def test_a_player_with_no_shot_row_is_graded_wrong_not_void():
    """Deliberate: the feed cannot separate 'did not play' from 'played, never
    shot'. Taking the harsher reading can only understate the record, and expected
    minutes are part of what the model claims."""
    e = {"market": "goal", "player": "X", "team": "T", "tainted": False}
    g = picks.grade_prop(e, None)
    assert g["graded"] == "wrong" and g["void"] is False and g["actual"] is None


def test_a_missing_confirmed_lineup_does_not_silently_kill_the_pick():
    """A confirmed XI is a signal, not a precondition.

    Gating the LOCK on both confirmed XIs read as rigour and worked as a kill
    switch: XIs land ~60 min before kickoff, the lock is at 45 min, and news.json
    is filled in by hand -- so unless someone was at the keyboard inside that
    15-minute window, no player pick could ever enter the record. The board would
    show provisional picks that vanished at lock time, and the Grades tab would sit
    permanently empty with nothing explaining why.

    The XI still does real work: it overrides appearance probability inside
    match_props, and every pick publishes `lineup_confirmed` so the two tiers stay
    distinguishable in the record."""
    assert _player_pick_publishable(3.0, lineup_ready=False) is True
    assert _player_pick_publishable(0.5, lineup_ready=True) is True
    assert _player_pick_publishable(0.5, lineup_ready=False) is True


# ------------------------------------------------- per-match actuals (grading)
class _FakeUnderstat:
    def __init__(self, frame):
        self._frame = frame

    def read_shot_events(self):
        return self._frame


def _events(rows):
    return pd.DataFrame(rows, columns=["game_id", "date", "team", "player",
                                       "result", "situation"])


def test_match_player_stats_counts_penalties_as_goals(monkeypatch):
    """A penalty is a shot with result 'Goal' and a NULL situation. It must count:
    an anytime-scorer pick wins on a penalty, and the rates feed's np_goals -- which
    excludes them -- would grade that winning pick as a miss."""
    ev = _events([
        (1, "2026-08-22 14:00", "Man City", "Haaland", "Goal", None),      # penalty
        (1, "2026-08-22 14:00", "Man City", "Haaland", "Saved Shot", "Open Play"),
    ])
    monkeypatch.setattr(players.sd, "Understat", lambda **kw: _FakeUnderstat(ev))
    monkeypatch.setattr(players, "canonical", lambda t, lg: t)
    out = players.match_player_stats("PL")
    row = out.iloc[0]
    assert row["goals"] == 1 and row["shots"] == 2 and row["sot"] == 2


def test_match_player_stats_excludes_own_goals(monkeypatch):
    """An own goal is credited to the scorer but is not a shot for his own team and
    never settles a scorer pick."""
    ev = _events([
        (1, "2026-08-22 14:00", "Everton", "Keane", "Own Goal", "Open Play"),
        (1, "2026-08-22 14:00", "Everton", "Keane", "Missed Shot", "Open Play"),
    ])
    monkeypatch.setattr(players.sd, "Understat", lambda **kw: _FakeUnderstat(ev))
    monkeypatch.setattr(players, "canonical", lambda t, lg: t)
    out = players.match_player_stats("PL")
    row = out.iloc[0]
    assert row["goals"] == 0 and row["shots"] == 1 and row["sot"] == 0


def test_unreadable_shot_feed_returns_empty_not_an_exception(monkeypatch):
    """Bundesliga's shot events crash upstream. Callers must get an empty frame and
    leave those picks ungraded, rather than the publish aborting or -- far worse --
    every pick being graded 'wrong' against data that simply is not there."""
    def boom(**kw):
        raise AttributeError("'list' object has no attribute 'values'")
    monkeypatch.setattr(players.sd, "Understat", boom)
    out = players.match_player_stats("BUNDESLIGA")
    assert out.empty and list(out.columns) == ["date", "game_id", "team", "player",
                                               "goals", "shots", "sot"]


def test_lineup_knowledge_is_frozen_with_the_pick():
    """The record must remember whether the XI was known when the pick was made.

    Without this the Grades tab pools picks frozen on a confirmed teamsheet with
    picks frozen on a guess, and a mediocre hit rate cannot be attributed to either
    the model or the missing team news -- two problems needing opposite responses.
    A review caught that `lineup_confirmed` was published on the card but never
    passed into lock_prop, so the graded record never saw it.
    """
    ko = pd.Timestamp("2026-08-22T14:00:00Z")
    log = {}
    picks.lock_prop(log, "k1", market="goal", player="Known", team="T",
                    p_pick=0.45, confidence=2, kickoff=ko,
                    now=ko - pd.Timedelta(minutes=30), lineup_confirmed=True)
    picks.lock_prop(log, "k2", market="goal", player="Guessed", team="T",
                    p_pick=0.45, confidence=2, kickoff=ko,
                    now=ko - pd.Timedelta(minutes=30), lineup_confirmed=False)
    assert log["k1"]["lineup_confirmed"] is True
    assert log["k2"]["lineup_confirmed"] is False

    # and the two tiers grade apart: same probability, opposite outcomes
    got = picks.grade_prop(log["k1"], {"goals": 1})
    missed = picks.grade_prop(log["k2"], {"goals": 0})
    confirmed = picks.record([g for g in (got, missed)
                              if g.get("lineup_confirmed") is True])
    unconfirmed = picks.record([g for g in (got, missed)
                                if g.get("lineup_confirmed") is not True])
    assert confirmed["correct"] == 1 and confirmed["wrong"] == 0
    assert unconfirmed["correct"] == 0 and unconfirmed["wrong"] == 1


def test_lock_prop_freezes_the_full_audit_trail():
    """A settled pick must be auditable, not just re-displayable. p_pick and the
    result alone cannot distinguish a 90%-appearance player who was oddly rested
    from a 50%-appearance player who simply didn't play -- the two need different
    follow-up, and only the frozen inputs can tell them apart."""
    ko = pd.Timestamp("2026-08-22T14:00:00Z")
    e = picks.lock_prop({}, "k", market="shots", player="Kane", team="Bayern Munich",
                        p_pick=0.78, confidence=5, kickoff=ko,
                        now=ko - pd.Timedelta(minutes=30),
                        appearance_pct=88.9, expected_minutes=68.4,
                        news_checked_hours_ago=1.5, doubt=False,
                        unavailable=False, team_attribution="Bayern Munich")
    assert e["appearance_pct"] == 88.9
    assert e["expected_minutes"] == 68.4
    assert e["news_checked_hours_ago"] == 1.5
    assert e["doubt"] is False
    assert e["unavailable"] is False
    assert e["team_attribution"] == "Bayern Munich"


def test_lock_prop_audit_fields_are_optional():
    """A data gap in any of these must never block a pick from locking -- the
    prediction itself does not depend on being able to explain itself."""
    ko = pd.Timestamp("2026-08-22T14:00:00Z")
    e = picks.lock_prop({}, "k", market="goal", player="X", team="T",
                        p_pick=0.45, confidence=2, kickoff=ko,
                        now=ko - pd.Timedelta(minutes=30))
    assert e["appearance_pct"] is None
    assert e["expected_minutes"] is None
    assert e["news_checked_hours_ago"] is None
    assert e["doubt"] is None
    assert e["unavailable"] is None
    assert e["team_attribution"] is None
