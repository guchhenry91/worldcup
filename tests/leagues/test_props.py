import pytest
import numpy as np
import pandas as pd

from leagues import props
from leagues.props import player_rates, match_props, GOAL_PRIORS, SHOT_PRIORS


def _logs():
    """Two players: one with a big sample, one with a single match."""
    rows = []
    for i in range(30):                      # 30 x 90min, ~15 goals -> 0.5 np/90
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "team": "Manchester City", "player": "Regular", "pos": "FW",
                     "minutes": 90, "np_goals": i % 2, "shots": 4, "sot": 2,
                     "npxg": 0.5, "pens_att": 0})
    rows.append({"date": pd.Timestamp("2026-02-01"), "team": "Manchester City",
                 "player": "NewSigning", "pos": "FW", "minutes": 90,
                 "np_goals": 1, "shots": 1, "sot": 1, "npxg": 0.1, "pens_att": 0})
    return pd.DataFrame(rows)


def test_low_sample_player_is_shrunk_toward_the_prior():
    r = player_rates(_logs(), ref=pd.Timestamp("2026-03-01"))
    new = r.loc[r["player"] == "NewSigning"].iloc[0]
    reg = r.loc[r["player"] == "Regular"].iloc[0]
    prior = GOAL_PRIORS["FW"]
    # raw rate for the new signing is nonsense (1 goal in 1 game); shrinkage must
    # pull it most of the way back to the FW prior.
    assert abs(new["rate90"] - prior) < abs(1.0 - prior) / 2
    assert abs(new["rate90"] - prior) < abs(reg["rate90"] - prior) + 0.25


def test_regular_player_keeps_his_own_signal():
    r = player_rates(_logs(), ref=pd.Timestamp("2026-03-01"))
    reg = r.loc[r["player"] == "Regular"].iloc[0]
    lo, hi = min(GOAL_PRIORS["FW"], 0.5), max(GOAL_PRIORS["FW"], 0.5)
    assert lo - 0.05 <= reg["rate90"] <= hi + 0.05


def test_shot_rates_present_and_shrunk_the_same_way():
    r = player_rates(_logs(), ref=pd.Timestamp("2026-03-01"))
    new = r.loc[r["player"] == "NewSigning"].iloc[0]
    assert "shots90" in r.columns and "sot_ratio" in r.columns
    # one match at 1 shot; must be pulled UP toward the ~2.5 FW shot prior
    assert new["shots90"] > 1.0
    assert 0.0 <= new["sot_ratio"] <= 1.0


from leagues.props import match_props


def _rates():
    return pd.DataFrame([
        {"team": "Manchester City", "player": "Haaland", "pos": "FW", "nineties": 30,
         "rate90": 0.80, "shots90": 4.0, "sot_ratio": 0.5},
        {"team": "Manchester City", "player": "Foden", "pos": "AM", "nineties": 25,
         "rate90": 0.30, "shots90": 2.0, "sot_ratio": 0.4},
        {"team": "Manchester City", "player": "Dias", "pos": "DF", "nineties": 30,
         "rate90": 0.05, "shots90": 0.5, "sot_ratio": 0.3},
        {"team": "Brentford", "player": "Mbeumo", "pos": "FW", "nineties": 28,
         "rate90": 0.40, "shots90": 2.5, "sot_ratio": 0.4},
        {"team": "Brentford", "player": "Wissa", "pos": "FW", "nineties": 20,
         "rate90": 0.35, "shots90": 2.0, "sot_ratio": 0.35},
    ])


def test_lambda_sum_equals_team_lambda():
    """THE invariant: the player model never disagrees with the match model."""
    props = match_props(_rates(), home="Manchester City", away="Brentford",
                        lam_home=2.1, lam_away=0.9,
                        minutes={}, pen_taker={"Manchester City": "Haaland"},
                        opp_shot_factor={"Manchester City": 1.0, "Brentford": 1.0})
    home = [p for p in props if p["team"] == "Manchester City"]
    assert abs(sum(p["lambda_goals"] for p in home) - 2.1) < 1e-6
    away = [p for p in props if p["team"] == "Brentford"]
    assert abs(sum(p["lambda_goals"] for p in away) - 0.9) < 1e-6


def test_anytime_is_poisson_of_lambda():
    props = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                        {}, {}, {})
    h = next(p for p in props if p["player"] == "Haaland")
    assert abs(h["anytime_pct"] - 100 * (1 - np.exp(-h["lambda_goals"]))) < 0.06


def test_opponent_matters():
    """Same player, tougher opponent (lower team lambda) => lower anytime %."""
    easy = match_props(_rates(), "Manchester City", "Brentford", 2.4, 0.9, {}, {}, {})
    hard = match_props(_rates(), "Manchester City", "Brentford", 1.2, 0.9, {}, {}, {})
    he = next(p for p in easy if p["player"] == "Haaland")["anytime_pct"]
    hh = next(p for p in hard if p["player"] == "Haaland")["anytime_pct"]
    assert hh < he - 5


def test_only_the_designated_taker_carries_penalty_lambda():
    props = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                        minutes={}, pen_taker={"Manchester City": "Haaland"},
                        opp_shot_factor={}, exp_pens={"Manchester City": 0.15})
    h = next(p for p in props if p["player"] == "Haaland")
    f = next(p for p in props if p["player"] == "Foden")
    assert h["penalty_taker"] is True
    assert f["penalty_taker"] is False
    home = [p for p in props if p["team"] == "Manchester City"]
    assert abs(sum(p["lambda_goals"] for p in home) - 2.1) < 1e-6


def test_player_ruled_out_is_removed_and_his_goals_redistribute():
    """A striker confirmed out must vanish from the props -- and his expected goals
    must go to his team-mates, not disappear: the team's players still have to sum
    to the match model's team lambda."""
    props = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                        unavailable={"Haaland"})
    names = [p["player"] for p in props if p["team"] == "Manchester City"]
    assert "Haaland" not in names
    assert "Foden" in names and "Dias" in names
    home = [p for p in props if p["team"] == "Manchester City"]
    assert abs(sum(p["lambda_goals"] for p in home) - 2.1) < 1e-6   # invariant holds


def test_doubtful_player_is_flagged_and_downweighted():
    full = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9)
    hurt = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                       doubtful={"Haaland"})
    h_full = next(p for p in full if p["player"] == "Haaland")
    h_hurt = next(p for p in hurt if p["player"] == "Haaland")
    assert h_hurt["doubt"] is True and h_full["doubt"] is False
    assert h_hurt["anytime_pct"] < h_full["anytime_pct"]     # fewer expected minutes
    # the team still sums to lambda -- the doubt reallocates, it does not leak
    home = [p for p in hurt if p["team"] == "Manchester City"]
    assert abs(sum(p["lambda_goals"] for p in home) - 2.1) < 1e-6


def test_removing_every_player_does_not_crash_or_invent_goals():
    props = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                        unavailable={"Haaland", "Foden", "Dias"})
    assert [p for p in props if p["team"] == "Manchester City"] == []
    assert [p for p in props if p["team"] == "Brentford"]        # other side intact


def test_appearance_probability_is_a_zero_inflated_event_model():
    """50% of 80 minutes is not the same distribution as certain for 40 minutes.

    The former must explicitly carry the chance the player never enters the game,
    especially for multi-event markets such as 2+ shots.
    """
    uncertain = match_props(
        _rates(), "Manchester City", "Brentford", 2.1, 0.9,
        playing_time={"Haaland": {
            "appearance_prob": 0.5, "minutes_if_playing": 80,
            "expected_minutes": 40}})
    certain = match_props(
        _rates(), "Manchester City", "Brentford", 2.1, 0.9,
        playing_time={"Haaland": {
            "appearance_prob": 1.0, "minutes_if_playing": 40,
            "expected_minutes": 40}})
    u = next(p for p in uncertain if p["player"] == "Haaland")
    c = next(p for p in certain if p["player"] == "Haaland")
    assert u["appearance_pct"] == 50.0
    assert u["expected_minutes"] == c["expected_minutes"] == 40.0
    assert u["p_shots_2plus"] != c["p_shots_2plus"]


def test_confirmed_lineup_overrides_historical_availability():
    pt = {"Haaland": {"appearance_prob": 0.25, "minutes_if_playing": 60,
                       "expected_minutes": 15}}
    provisional = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                              playing_time=pt)
    confirmed = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                            playing_time=pt, confirmed_starters={"Haaland"})
    p = next(x for x in provisional if x["player"] == "Haaland")
    c = next(x for x in confirmed if x["player"] == "Haaland")
    assert p["appearance_pct"] == 25.0
    assert c["appearance_pct"] == 100.0
    assert c["expected_minutes"] >= 70.0
    assert c["p_shots_2plus"] > p["p_shots_2plus"]


def test_confirmed_bench_caps_minutes_and_appearance_chance():
    bench = match_props(
        _rates(), "Manchester City", "Brentford", 2.1, 0.9,
        playing_time={"Haaland": {"appearance_prob": 0.95,
                                  "minutes_if_playing": 85,
                                  "expected_minutes": 80.75}},
        confirmed_bench={"Haaland"})
    h = next(x for x in bench if x["player"] == "Haaland")
    assert h["appearance_pct"] == 35.0
    assert h["expected_minutes"] <= 8.8


# ------------------------------------------------------- absence -> team lambda
def _abs_rates():
    return pd.DataFrame(
        [{"team": "A", "player": f"A{i}", "pos": "FW", "nineties": 10.0,
          "rate90": 0.5, "shots90": s, "sot_ratio": 0.35}
         for i, s in enumerate([4.0, 2.0, 1.5, 1.0, 1.0, 0.5])])


def test_absence_penalty_scales_with_share_of_the_team_shooting():
    r = _abs_rates()
    top = props.absence_penalty(r, "A", {"A0"})     # 4 of 10 shots
    small = props.absence_penalty(r, "A", {"A5"})   # 0.5 of 10
    assert top > small > 0
    assert top == pytest.approx(props.ABSENCE_GOAL_COST * 0.4)


def test_absence_penalty_is_conservative_not_catastrophic():
    """The measured effect is an UPPER bound (the absence proxy still catches some
    players dropped for form), and the CI is wide, so the shipped constant sits
    well below the point estimate. Over-reacting would swing a win probability by
    ten points on one absence and make us worse than the market we already trail."""
    r = _abs_rates()
    # even losing the entire attack cannot wipe out a team's expected goals
    everyone = props.absence_penalty(r, "A", set(r["player"]))
    assert everyone == pytest.approx(props.ABSENCE_GOAL_COST)
    assert props.ABSENCE_GOAL_COST < 0.77          # below the measured estimate
    assert 1.6 - everyone > 1.0                    # a strong side stays strong


def test_absence_penalty_returns_zero_when_it_cannot_be_computed():
    """A data gap must never silently move a published prediction."""
    r = _abs_rates()
    assert props.absence_penalty(r, "A", set()) == 0.0
    assert props.absence_penalty(r, "Unknown Team", {"A0"}) == 0.0
    assert props.absence_penalty(pd.DataFrame(columns=r.columns), "A", {"A0"}) == 0.0


def test_doubtful_absence_is_weighted_not_ignored_or_full():
    """Matches the props board's own treatment of the same fact (DOUBT_MINUTES_
    FACTOR): a doubtful player counts for a fraction of a confirmed-out one, not
    zero (the match model used to ignore doubtful players entirely) and not the
    full penalty (that would be treating "doubtful" as "definitely out")."""
    r = _abs_rates()
    out = props.absence_penalty(r, "A", {"A0"})                  # confirmed out
    doubt = props.absence_penalty(r, "A", set(), {"A0"})          # doubtful only
    none = props.absence_penalty(r, "A", set(), set())
    assert none == 0.0
    assert 0.0 < doubt < out
    assert doubt == pytest.approx(out * props.DOUBTFUL_ABSENCE_WEIGHT)


def test_doubtful_and_confirmed_out_combine_without_double_counting():
    r = _abs_rates()
    # A0 confirmed out, A1 doubtful -- if a name is passed as both, confirmed
    # status must win rather than the doubt weight silently discounting it.
    combined = props.absence_penalty(r, "A", {"A0"}, {"A0", "A1"})
    out_and_doubt_a1 = (props.absence_penalty(r, "A", {"A0"}) +
                        props.absence_penalty(r, "A", set(), {"A1"}))
    assert combined == pytest.approx(out_and_doubt_a1)


def test_unmodeled_absentee_positions_flags_defenders_and_keepers_only():
    r = pd.DataFrame([
        {"team": "A", "player": "Striker", "pos": "FW", "nineties": 10.0,
         "rate90": 0.5, "shots90": 3.0, "sot_ratio": 0.35},
        {"team": "A", "player": "CenterBack", "pos": "DF", "nineties": 10.0,
         "rate90": 0.02, "shots90": 0.2, "sot_ratio": 0.1},
        {"team": "A", "player": "Keeper", "pos": "GK", "nineties": 10.0,
         "rate90": 0.0, "shots90": 0.0, "sot_ratio": 0.0},
    ])
    # An attacker being out is priced in by absence_penalty -- not flagged.
    assert props.unmodeled_absentee_positions(r, "A", {"Striker"}) == []
    # Defenders/keepers are invisible to the shot-share mechanism -- flagged.
    assert props.unmodeled_absentee_positions(r, "A", {"CenterBack"}) == ["CenterBack"]
    assert props.unmodeled_absentee_positions(r, "A", {"Keeper"}) == ["Keeper"]
    both = props.unmodeled_absentee_positions(r, "A", {"CenterBack", "Keeper", "Striker"})
    assert both == ["CenterBack", "Keeper"]
    assert props.unmodeled_absentee_positions(r, "A", set()) == []
