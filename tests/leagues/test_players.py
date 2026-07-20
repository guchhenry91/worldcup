import pandas as pd
import pytest

from leagues.names import UnknownTeam
from leagues.players import (build_player_logs, reconcile_rates_to_roster,
                             season_end, understat_position)


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


def test_expected_minutes_are_per_match_not_per_season():
    from leagues.players import expected_minutes
    df = build_player_logs(_season_stats(), _shots(), "PL")
    em = expected_minutes(df)
    # Keeper: 3420 minutes over a 38-match season -> a full 90 every week
    assert em["Keeper"] == 90.0
    # Mover: 2300 minutes in his latest season -> ~60 per match, NOT 90
    assert 55 < em["Mover"] < 65


def test_playing_time_separates_appearance_from_conditional_minutes():
    from leagues.players import playing_time
    df = build_player_logs(_season_stats(), _shots(), "PL")
    pt = playing_time(df)
    mover = pt["Mover"]
    assert 0 < mover["appearance_prob"] < 1
    assert mover["minutes_if_playing"] > mover["expected_minutes"]
    assert abs(mover["appearance_prob"] * mover["minutes_if_playing"]
               - mover["expected_minutes"]) < 1e-9


def test_playing_time_respects_shorter_league_seasons():
    from leagues.players import playing_time
    df = pd.DataFrame([{
        "season": "2526", "player": "Regular", "minutes": 2700,
        "appearances": 34,
    }])
    pt = playing_time(df, matches_per_season=34)["Regular"]
    assert pt["appearance_prob"] > 0.95
    assert 78 < pt["minutes_if_playing"] < 81


def test_only_explicit_complete_lineups_are_confirmed():
    from leagues.players import lineup_players, lineups_confirmed
    news = {
        "A": {"lineup_confirmed": True,
              "starters": [f"A{i}" for i in range(11)], "bench": ["A12"]},
        "B": {"lineup_confirmed": False,
              "starters": [f"B{i}" for i in range(11)]},
    }
    starters, bench = lineup_players(news, ("A", "B"))
    assert "A0" in starters and "A12" in bench
    assert "B0" not in starters
    assert lineups_confirmed(news, ("A", "B")) is False
    news["B"]["lineup_confirmed"] = True
    assert lineups_confirmed(news, ("A", "B")) is True


def test_current_squad_excludes_players_who_did_not_appear_last_season():
    """Five seasons of departed players would otherwise share out the team's
    expected goals and crush the real strikers to a few percent."""
    from leagues.players import current_squad
    stats = _season_stats()
    stats = pd.concat([stats, pd.DataFrame([
        {"season": "2223", "team": "Man City", "player": "LongGone", "position": "F S",
         "matches": 20, "minutes": 1500, "np_goals": 8, "np_xg": 7.0, "shots": 40},
    ])], ignore_index=True)
    df = build_player_logs(stats, _shots(), "PL")
    squad = current_squad(df)
    assert "Mover" in squad and "Keeper" in squad
    assert "LongGone" not in squad


def test_missing_shot_events_degrades_instead_of_crashing():
    """If shot-level data is unavailable (upstream parser bug on some leagues),
    logs still build from season stats: SOT falls back to the league-average
    ratio and penalty attempts are zero, rather than the whole league failing."""
    stats = _season_stats()
    df = build_player_logs(stats, None, "PL")          # shots=None
    assert len(df) == 3
    row = df[(df["player"] == "Mover") & (df["season"] == "2526")].iloc[0]
    # 85 shots at the ~0.35 league on-target prior -> ~30, and no penalties known
    assert row["pens_att"] == 0
    assert 0 < row["sot"] <= row["shots"]


def test_transfer_override_moves_and_removes_players():
    """Summer-window moves aren't in last season's data, so an override layer
    re-attributes a moved player to his new club and drops one who left."""
    stats = _season_stats()   # Mover (Man City), Keeper (Man City)
    # add a player at Brentford who "left" for another league this window
    import pandas as pd
    stats = pd.concat([stats, pd.DataFrame([
        {"season": "2526", "team": "Brentford", "player": "Leaver", "position": "F S",
         "matches": 30, "minutes": 2500, "np_goals": 18, "np_xg": 15.0, "shots": 90},
    ])], ignore_index=True)
    transfers = {"Mover": "Arsenal", "Leaver": None}   # Mover -> Arsenal; Leaver gone
    df = build_player_logs(stats, _shots(), "PL", transfers=transfers)
    assert set(df[df["player"] == "Mover"]["team"]) == {"Arsenal"}   # reattributed
    assert "Leaver" not in set(df["player"])                          # removed
    assert "Keeper" in set(df["player"])                             # untouched


def test_roster_reconciliation_reassigns_known_players_and_fails_closed(
        monkeypatch):
    rates = pd.DataFrame([
        {"team": "Old Club", "player": "Álex One", "rate90": 0.4},
        {"team": "Old Club", "player": "Departed", "rate90": 0.3},
        {"team": "Thin Club", "player": "Thin Player", "rate90": 0.2},
    ])
    complete = [{"id": str(i), "name": f"Squad {i}"} for i in range(17)]
    complete.append({"id": "99", "name": "Alex One"})
    snapshot = {
        "New Club": {"players": complete},
        "Thin Club": {"players": [{"id": "1", "name": "Thin Player"}]},
    }
    monkeypatch.setattr("leagues.players.load_roster_snapshot",
                        lambda league: snapshot)
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda: 1.0)

    safe, incomplete, unmatched = reconcile_rates_to_roster(rates, "PL")

    assert safe[["team", "player"]].to_dict("records") == [
        {"team": "New Club", "player": "Álex One"}]
    assert incomplete == ["Thin Club"]
    assert unmatched == ["Old Club/Departed", "Thin Club/Thin Player"]


def test_missing_roster_snapshot_withholds_all_player_rates(monkeypatch):
    rates = pd.DataFrame([
        {"team": "Arsenal", "player": "Player", "rate90": 0.4},
    ])
    monkeypatch.setattr("leagues.players.load_roster_snapshot", lambda league: {})
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda: None)
    safe, incomplete, unmatched = reconcile_rates_to_roster(rates, "PL")
    assert safe.empty
    assert incomplete == ["Arsenal"]
    assert unmatched == []


def test_stale_roster_snapshot_withholds_all_player_rates(monkeypatch):
    rates = pd.DataFrame([
        {"team": "Arsenal", "player": "Player", "rate90": 0.4},
    ])
    snapshot = {
        "Arsenal": {"players": [
            {"id": str(i), "name": "Player" if i == 0 else f"Squad {i}"}
            for i in range(18)
        ]}
    }
    monkeypatch.setattr("leagues.players.load_roster_snapshot",
                        lambda league: snapshot)
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours",
                        lambda: 72.01)
    safe, incomplete, _ = reconcile_rates_to_roster(rates, "PL")
    assert safe.empty
    assert incomplete == ["Arsenal"]
