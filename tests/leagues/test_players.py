import pandas as pd
import pytest

from leagues.names import UnknownTeam
from leagues.players import build_player_logs, season_end, understat_position


def _season_stats():
    return pd.DataFrame([
        # a striker who moved clubs between seasons
        {"season": "2425", "team": "Brentford", "player": "Mover", "position": "F S",
         "matches": 30, "minutes": 2400, "np_goals": 12, "np_xg": 10.5, "shots": 70},
        {"season": "2526", "team": "Man City", "player": "Mover", "position": "F S",
         "matches": 28, "minutes": 2300, "np_goals": 15, "np_xg": 14.0, "shots": 85},
        {"season": "2526", "team": "Man City", "player": "Keeper", "position": "GK",
         "matches": 38, "minutes": 3420, "np_goals": 0, "np_xg": 0.0, "shots": 0},
    ])


def _shots():
    return pd.DataFrame([
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": "Open Play", "result": "Goal"},
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": "Open Play", "result": "Saved Shot"},
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": "Open Play", "result": "Missed Shot"},
        # soccerdata maps Understat's "Penalty" situation to NA -- see players.py
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": None, "result": "Goal"},
        {"season": "2425", "team": "Brentford", "player": "Mover",
         "situation": "Open Play", "result": "Blocked Shot"},
    ])


def test_one_row_per_player_season_with_canonical_teams():
    df = build_player_logs(_season_stats(), _shots(), "PL")
    assert len(df) == 3
    assert set(df["team"]) == {"Manchester City"}     # canonical, and see below


def test_a_transferred_player_is_attributed_to_his_CURRENT_club():
    """Both of Mover's seasons must carry his 2526 club, or player_rates would
    split him into two half-players at two different clubs."""
    df = build_player_logs(_season_stats(), _shots(), "PL")
    mover = df[df["player"] == "Mover"]
    assert len(mover) == 2
    assert set(mover["team"]) == {"Manchester City"}   # not Brentford


def test_shots_on_target_and_penalties_come_from_shot_events():
    df = build_player_logs(_season_stats(), _shots(), "PL")
    row = df[(df["player"] == "Mover") & (df["season"] == "2526")].iloc[0]
    # Goal + Saved Shot + the scored penalty. Penalties count: Understat's `shots`
    # total includes them, so the on-target ratio must too or sot/shots is skewed.
    # (Missed and Blocked are not on target.)
    assert row["sot"] == 3
    assert row["pens_att"] == 1


def test_position_is_mapped_off_understats_first_real_token():
    assert understat_position("F S") == "FW"
    assert understat_position("D S") == "DF"
    assert understat_position("M S") == "MF"
    assert understat_position("GK") == "GK"
    assert understat_position("S") == "MF"      # sub-only: fall back to MF


def test_season_end_dates_drive_the_decay():
    assert season_end("2526") == pd.Timestamp("2026-05-31")
    assert season_end("2122") == pd.Timestamp("2022-05-31")


def test_unmapped_team_fails_loudly():
    stats = _season_stats()
    stats.loc[0, "team"] = "Wimbledon FC"
    with pytest.raises(UnknownTeam):
        build_player_logs(stats, _shots(), "PL")


def test_penalty_taker_is_derived_from_na_situation_shots():
    """Regression guard: soccerdata maps Understat's "Penalty" situation to NA.
    Matching the string "penalty" finds nothing and every club silently ends up
    with no penalty taker."""
    from leagues.players import penalty_takers
    df = build_player_logs(_season_stats(), _shots(), "PL")
    assert df[df["player"] == "Mover"]["pens_att"].sum() == 1
    assert penalty_takers(df)["Manchester City"] == "Mover"
