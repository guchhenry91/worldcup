import pandas as pd

from leagues.sim import final_table, rank_teams


def test_tiebreakers_points_then_gd_then_gf():
    rows = [
        {"team": "A", "points": 10, "gd": 5, "gf": 12},
        {"team": "B", "points": 10, "gd": 5, "gf": 15},   # same pts+gd, more GF -> above A
        {"team": "C", "points": 11, "gd": 0, "gf": 3},    # more points -> top
        {"team": "D", "points": 10, "gd": 2, "gf": 20},   # worse gd -> below A and B
    ]
    assert rank_teams(pd.DataFrame(rows)) == ["C", "B", "A", "D"]


def test_played_results_are_locked_in():
    """A team that already won 3-0 keeps those points in every simulation."""
    played = pd.DataFrame([
        {"home": "A", "away": "B", "home_goals": 3, "away_goals": 0, "played": True},
    ])
    remaining = pd.DataFrame(columns=["home", "away"])
    table = final_table(played, remaining, sample=lambda h, a: (0, 0)).set_index("team")
    assert table.loc["A", "points"] == 3 and table.loc["A", "gd"] == 3
    assert table.loc["B", "points"] == 0 and table.loc["B", "gd"] == -3


def test_remaining_fixtures_are_sampled():
    played = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals", "played"])
    remaining = pd.DataFrame([{"home": "A", "away": "B"}])
    table = final_table(played, remaining, sample=lambda h, a: (2, 1)).set_index("team")
    assert table.loc["A", "points"] == 3 and table.loc["A", "gf"] == 2


import numpy as np
from leagues.sim import order_teams


def _one_sim(pts, gd, gf):
    """Shape [1, T] arrays for a single simulated season."""
    return (np.array([pts], dtype=np.int32), np.array([gd], dtype=np.int32),
            np.array([gf], dtype=np.int32))


def _h2h(T, results):
    """h2h_pts/gd/gf [T,T,1] from a list of (home, away, hg, ag) meetings."""
    hp = np.zeros((T, T, 1), dtype=np.int32)
    hg = np.zeros((T, T, 1), dtype=np.int32)
    hf = np.zeros((T, T, 1), dtype=np.int32)
    for (i, j, gi, gj) in results:
        hp[i, j, 0] += 3 if gi > gj else (1 if gi == gj else 0)
        hp[j, i, 0] += 3 if gj > gi else (1 if gi == gj else 0)
        hg[i, j, 0] += gi - gj; hg[j, i, 0] += gj - gi
        hf[i, j, 0] += gi; hf[j, i, 0] += gj
    return hp, hg, hf


def test_gd_league_breaks_ties_on_goal_difference():
    """PL/Bundesliga/Ligue 1: points -> goal difference -> goals for."""
    pts, gd, gf = _one_sim([70, 70], [20, 5], [60, 60])
    hp, hg, hf = _h2h(2, [(1, 0, 4, 0)])      # team 0 LOST the h2h badly...
    order = order_teams(pts, gd, gf, hp, hg, hf, "gd")
    assert list(order[0]) == [0, 1]           # ...but its better GD still wins


def test_h2h_league_breaks_ties_on_head_to_head_first():
    """La Liga: the club with the worse overall GD still finishes above if it won
    the head-to-head."""
    pts, gd, gf = _one_sim([70, 70], [20, 5], [60, 60])
    hp, hg, hf = _h2h(2, [(1, 0, 4, 0)])      # team 1 beat team 0 4-0
    order = order_teams(pts, gd, gf, hp, hg, hf, "h2h")
    assert list(order[0]) == [1, 0]           # team 1 goes above despite worse GD


def test_h2h_falls_back_to_goal_difference_when_head_to_head_is_level():
    pts, gd, gf = _one_sim([70, 70], [20, 5], [60, 60])
    hp, hg, hf = _h2h(2, [(0, 1, 1, 1)])      # h2h level (a draw)
    order = order_teams(pts, gd, gf, hp, hg, hf, "h2h")
    assert list(order[0]) == [0, 1]           # better overall GD decides


def test_teams_not_tied_on_points_are_never_reordered_by_h2h():
    pts, gd, gf = _one_sim([71, 70], [0, 30], [50, 60])
    hp, hg, hf = _h2h(2, [(1, 0, 5, 0)])      # team 0 lost h2h, worse GD
    order = order_teams(pts, gd, gf, hp, hg, hf, "h2h")
    assert list(order[0]) == [0, 1]           # more points wins outright


def test_h2h_three_team_noncyclic_uses_the_full_mini_league():
    """The bug an adjacent-pairwise pass gets wrong: A beat B, B beat C, A DREW C
    (no cycle). Mini-league: A=4, B=3, C=1 -> [A, B, C]. A must top the group even
    though A's direct meeting with C was level."""
    # A,B,C (0,1,2) all level on points; D (3) lower. Overall GD would rank C top
    # of the group (best gd), so a correct result must come from the mini-league.
    pts, gd, gf = _one_sim([60, 60, 60, 40], [0, 0, 10, -30], [40, 40, 50, 20])
    hp, hg, hf = _h2h(4, [
        (0, 1, 3, 0),   # A beat B 3-0
        (1, 2, 3, 0),   # B beat C 3-0
        (0, 2, 1, 1),   # A drew C 1-1
    ])
    order = order_teams(pts, gd, gf, hp, hg, hf, "h2h")
    assert list(order[0]) == [0, 1, 2, 3]     # A, B, C by mini-league; D last on points
