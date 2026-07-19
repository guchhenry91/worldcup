"""The cross-league boards must never quietly lose a graded pick.

These cover the two functions a code audit found carrying the worst defects and
having no tests at all. Both bugs were real and reproduced before being fixed:

  - a transient fetch failure rebuilt `settled` without one league, silently
    deleting its record -- and the deletion removes losses as readily as wins, so
    a five-second timeout flattered the model,
  - a pick inside the lock window was published TWICE, once from the payload and
    once from the log, the second copy with no scoreline.
"""
import pandas as pd
import pytest

from leagues import picks, publish


@pytest.fixture
def frozen_losses(tmp_path, monkeypatch):
    """One league with two frozen, high-confidence, LOSING picks."""
    (tmp_path / "pl").mkdir()
    log = {}
    for mid, ko in ((1, "2026-08-22T14:00:00Z"), (2, "2026-08-22T16:30:00Z")):
        picks.lock_pick(log, f"2026:{mid}", pick="Arsenal", confidence=5, kickoff=ko,
                        now=pd.Timestamp(ko) - pd.Timedelta(hours=3), p_pick=0.72)
    picks.save_log(log, tmp_path / "pl" / "picks_log.json")

    fx = pd.DataFrame([
        {"match_id": 1, "round": 1, "date": pd.Timestamp("2026-08-22T14:00:00"),
         "home": "Arsenal", "away": "Chelsea", "played": True,
         "home_goals": 0, "away_goals": 2},
        {"match_id": 2, "round": 1, "date": pd.Timestamp("2026-08-22T16:30:00"),
         "home": "Arsenal", "away": "Spurs", "played": True,
         "home_goals": 1, "away_goals": 3},
    ])
    monkeypatch.setattr(publish, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(publish, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(publish, "FILE_FOR", {"PL": "pl.json"})
    return fx


def test_both_losses_are_graded_when_the_feed_works(frozen_losses, monkeypatch):
    monkeypatch.setattr(publish.fixtures, "fetch_fixtures", lambda lg: frozen_losses)
    b = publish.build_best_picks()
    assert b["record"] == {"correct": 0, "wrong": 2, "total": 2, "void": 0,
                           "pending": 0, "by_confidence": {"5": {"correct": 0, "total": 2}}}
    assert b["_incomplete"] == []


def test_a_fetch_blip_refuses_the_board_instead_of_deleting_the_losses(
        frozen_losses, monkeypatch):
    """The regression that matters. Before the fix this returned a clean 0-0
    record and was published, erasing two losses on a network timeout."""
    def blip(lg):
        raise TimeoutError("connection timed out")
    monkeypatch.setattr(publish.fixtures, "fetch_fixtures", blip)
    b = publish.build_best_picks()
    # the record it WOULD have published is empty...
    assert b["record"]["total"] == 0
    # ...so it must be flagged, and main() must not write it
    assert b["_incomplete"] == ["Premier League"]


def test_a_locked_pick_appears_once_not_twice(frozen_losses, monkeypatch):
    """Inside the lock window the payload and the log both describe the fixture.
    Publishing both put a duplicate, blank-scored card on the board for every
    locked pick on matchday."""
    import json
    fx = frozen_losses.copy()
    fx["played"] = False                       # not yet played -> goes to `upcoming`
    monkeypatch.setattr(publish.fixtures, "fetch_fixtures", lambda lg: fx)
    (publish.OUT / "pl.json").write_text(json.dumps({"matches": [{
        "id": 1, "matchweek": 1, "date": "2026-08-22T14:00:00",
        "home": "Arsenal", "away": "Chelsea",
        "prediction": {"best_pick": True, "pick": "Arsenal", "confidence": 5,
                       "p_pick": 0.72, "score": "2-0", "provisional": False},
    }]}), encoding="utf-8")

    b = publish.build_best_picks()
    ids = [u["id"] for u in b["upcoming"]]
    assert ids.count(1) == 1, f"fixture 1 published {ids.count(1)} times"
    # and the surviving copy is the informative one
    kept = next(u for u in b["upcoming"] if u["id"] == 1)
    assert kept["score"] == "2-0"


def test_a_transient_player_feed_failure_does_not_claim_the_league_is_ungradeable(
        frozen_losses, monkeypatch):
    """Bundesliga genuinely has no shot feed. A one-off timeout elsewhere is a
    different thing, and must not publish the false claim that the league cannot
    be graded -- nor silently drop its record."""
    monkeypatch.setattr(publish.fixtures, "fetch_fixtures", lambda lg: frozen_losses)
    monkeypatch.setattr(publish.players, "match_player_stats",
                        lambda lg: (_ for _ in ()).throw(TimeoutError("blip")))
    monkeypatch.setattr(publish.players, "shot_events_available", lambda lg: True)
    (publish.PICKS_DIR / "pl" / "player_picks_log.json").write_text(
        '{"2026:1:goal:Kane": {"market": "goal", "player": "Kane", "team": "Arsenal",'
        ' "p_pick": 0.45, "confidence": 2, "tainted": false,'
        ' "kickoff": "2026-08-22T14:00:00+00:00"}}', encoding="utf-8")

    pp = publish.build_player_picks()
    assert pp["ungradeable_leagues"] == []      # NOT branded permanently ungradeable
    assert pp["_incomplete"] == ["Premier League"]
