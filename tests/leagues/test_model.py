import numpy as np
import pandas as pd
import pytest

from leagues.model import Calibrator, LeagueModel, blend_probs, dc_tau, scoreline_grid


def test_dc_tau_lifts_draws_and_trims_1_0():
    assert dc_tau(0, 0, 1.4, 1.2, rho=-0.1) > 1.0
    assert dc_tau(1, 1, 1.4, 1.2, rho=-0.1) > 1.0
    assert dc_tau(1, 0, 1.4, 1.2, rho=-0.1) < 1.0
    assert dc_tau(3, 2, 1.4, 1.2, rho=-0.1) == 1.0


def test_scoreline_grid_is_a_normalized_distribution():
    g = scoreline_grid(1.5, 1.1, rho=-0.1, max_goals=10)
    assert abs(g.sum() - 1.0) < 1e-9
    assert (g >= 0).all()


def test_grid_gives_higher_home_win_prob_for_stronger_home_team():
    strong = scoreline_grid(2.2, 0.8, rho=-0.1)
    weak = scoreline_grid(0.8, 2.2, rho=-0.1)
    assert np.tril(strong, -1).sum() > 0.5 > np.tril(weak, -1).sum()


def _toy_matches(n=300):
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 1.8, "B": 1.4, "C": 1.0, "D": 0.7}
    rows = []
    start = pd.Timestamp("2025-08-01")
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        hg = int(rng.poisson(strength[h] * 1.15))
        ag = int(rng.poisson(strength[a] * 0.85))
        rows.append({"date": start + pd.Timedelta(days=i), "home": h, "away": a,
                     "home_goals": hg, "away_goals": ag,
                     "home_xg": float(hg), "away_xg": float(ag)})
    return pd.DataFrame(rows)


def test_model_fits_and_ranks_teams_correctly():
    m = LeagueModel().fit(_toy_matches(), ref=pd.Timestamp("2026-06-20"))
    p = m.predict("A", "D")
    assert p["p_home"] > p["p_away"]
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1.0) < 1e-6
    assert 0 < p["p_draw"] < 0.45


def test_predict_unknown_team_raises():
    m = LeagueModel().fit(_toy_matches(), ref=pd.Timestamp("2026-06-20"))
    with pytest.raises(KeyError):
        m.predict("A", "ZZ")


def test_blend_probs_averages_and_normalizes():
    out = blend_probs((0.5, 0.3, 0.2), (0.7, 0.2, 0.1), weight=0.5)
    assert abs(sum(out) - 1.0) < 1e-9
    assert abs(out[0] - 0.6) < 1e-9


def test_calibrator_preserves_discrimination_and_normalizes():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([2, 1, 2], size=500)
    y = np.array([rng.choice(3, p=row) for row in p])
    cal = Calibrator().fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.corrcoef(out[:, 0], p[:, 0])[0, 1] > 0.9


def test_score_for_outcome_agrees_with_the_picked_result():
    """The displayed scoreline must not contradict the pick. 1-1 is often the
    single most likely EXACT score even when one side is clearly favoured, so the
    card must show the most likely score *within* the picked outcome."""
    import numpy as np
    from leagues.model import scoreline_grid, outcome_probs, score_for_outcome
    # a fixture where the away side is favoured overall
    grid = scoreline_grid(0.9, 1.8, rho=-0.05)
    ph, pd_, pa = outcome_probs(grid)
    assert pa > ph and pa > pd_                      # away win is the pick

    home_s = score_for_outcome(grid, "home")
    draw_s = score_for_outcome(grid, "draw")
    away_s = score_for_outcome(grid, "away")
    hh, ha = (int(x) for x in home_s.split("-"))
    dh, da = (int(x) for x in draw_s.split("-"))
    ah, aa = (int(x) for x in away_s.split("-"))
    assert hh > ha, f"home pick gave {home_s}"       # a home win scoreline
    assert dh == da, f"draw pick gave {draw_s}"      # a level scoreline
    assert aa > ah, f"away pick gave {away_s}"       # an away win scoreline


def test_top_scorelines_are_ranked_and_probabilities_sane():
    from leagues.model import scoreline_grid, top_scorelines
    grid = scoreline_grid(1.6, 1.1, rho=-0.05)
    top = top_scorelines(grid, n=3)
    assert len(top) == 3
    pcts = [t["pct"] for t in top]
    assert pcts == sorted(pcts, reverse=True)        # ranked most likely first
    assert all(0 < p < 100 for p in pcts)
    assert len({t["score"] for t in top}) == 3       # no duplicates
    # the top entry must be the true grid mode
    import numpy as np
    h, a = np.unravel_index(np.argmax(grid), grid.shape)
    assert top[0]["score"] == f"{int(h)}-{int(a)}"


def test_top_scorelines_probabilities_match_the_grid():
    """The displayed % must be the real cell probability, not a re-normalised one."""
    import numpy as np
    from leagues.model import scoreline_grid, top_scorelines
    grid = scoreline_grid(2.0, 0.7, rho=-0.03)
    top = top_scorelines(grid, n=3)
    for t in top:
        h, a = (int(x) for x in t["score"].split("-"))
        assert abs(t["pct"] - 100 * grid[h, a]) < 0.05


def test_goals_markets_derive_from_the_grid():
    """O/U 2.5 and BTTS are exact sums over the same scoreline grid, so they must
    be consistent with it and with each other."""
    import numpy as np
    from leagues.model import scoreline_grid, goals_markets
    g = scoreline_grid(1.7, 1.3, rho=-0.05)
    mk = goals_markets(g)
    # probabilities are complementary pairs
    assert abs(mk["p_over25"] + mk["p_under25"] - 1) < 1e-9
    assert abs(mk["p_btts"] + mk["p_btts_no"] - 1) < 1e-9
    assert all(0 < mk[k] < 1 for k in ("p_over25", "p_under25", "p_btts", "p_btts_no"))
    # cross-check against a direct sum over the grid. goals_markets rounds to 3dp
    # for the payload, so the tolerance is that rounding, not machine epsilon.
    I, J = np.indices(g.shape)
    assert abs(mk["p_over25"] - float(g[(I + J) > 2.5].sum())) < 5e-4
    assert abs(mk["p_btts"] - float(g[(I > 0) & (J > 0)].sum())) < 5e-4


def test_high_scoring_fixture_favours_over_and_btts():
    from leagues.model import scoreline_grid, goals_markets
    high = goals_markets(scoreline_grid(2.3, 1.9, rho=-0.05))
    low = goals_markets(scoreline_grid(0.7, 0.6, rho=-0.05))
    assert high["p_over25"] > 0.5 > low["p_over25"]
    assert high["p_btts"] > low["p_btts"]
    # the stated pick must follow the probability
    assert high["over_under_pick"] == "over" and low["over_under_pick"] == "under"
    assert high["btts_pick"] == "yes" and low["btts_pick"] == "no"
