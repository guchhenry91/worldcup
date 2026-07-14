"""Season Monte Carlo: sample every remaining fixture, tally the table, repeat.

PL tie-breakers, in order: points -> goal difference -> goals for. (Head-to-head
only separates teams still level on all three, which is rare enough that we leave
it to the alphabetical fallback rather than pretend to a precision the model
does not have.)
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from leagues import config
from leagues.model import scoreline_grid

N_SIMS = 10000


def rank_teams(table: pd.DataFrame) -> list[str]:
    """Order teams by points, then goal difference, then goals for."""
    t = table.sort_values(["points", "gd", "gf", "team"],
                          ascending=[False, False, False, True])
    return t["team"].tolist()


def final_table(played: pd.DataFrame, remaining: pd.DataFrame, sample) -> pd.DataFrame:
    """One simulated season. `sample(home, away) -> (home_goals, away_goals)`.

    Real results in `played` are locked in; only `remaining` is sampled.
    """
    pts, gf, ga = defaultdict(int), defaultdict(int), defaultdict(int)

    def record(h, a, hg, ag):
        gf[h] += hg; ga[h] += ag
        gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif ag > hg:
            pts[a] += 3
        else:
            pts[h] += 1
            pts[a] += 1

    for _, m in played.iterrows():
        record(m["home"], m["away"], int(m["home_goals"]), int(m["away_goals"]))
    for _, m in remaining.iterrows():
        hg, ag = sample(m["home"], m["away"])
        record(m["home"], m["away"], int(hg), int(ag))

    teams = sorted(set(gf) | set(ga) | set(pts))
    return pd.DataFrame([{"team": t, "points": pts[t], "gf": gf[t], "ga": ga[t],
                          "gd": gf[t] - ga[t]} for t in teams])


def _sampler(model, rng):
    """Draw a scoreline from the model's Dixon-Coles grid (cached per fixture)."""
    cache = {}

    def sample(home, away):
        key = (home, away)
        if key not in cache:
            lh, la = model.lambdas(home, away)
            grid = scoreline_grid(lh, la, model.rho)
            cache[key] = (grid.ravel(), grid.shape)
        flat, shape = cache[key]
        idx = rng.choice(len(flat), p=flat)
        return np.unravel_index(idx, shape)

    return sample


def simulate_season(model, played: pd.DataFrame, remaining: pd.DataFrame,
                    league: str = "PL", n: int = N_SIMS, seed: int = 7) -> pd.DataFrame:
    """Run n seasons; return the projected table with title/top-4/relegation %.

    Vectorized: every fixture is sampled for all n seasons at once and the tables
    are tallied in numpy. The obvious implementation -- call final_table() n times
    -- is ~40,000x slower here (10k seasons x 380 fixtures of pandas row access),
    which turned a publish into an hour-long job.
    """
    lg = config.get(league)
    rng = np.random.default_rng(seed)

    teams = sorted(set(played["home"]) | set(played["away"])
                   | set(remaining["home"]) | set(remaining["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    pts = np.zeros((n, T), dtype=np.int32)
    gf = np.zeros((n, T), dtype=np.int32)
    ga = np.zeros((n, T), dtype=np.int32)

    # results already in the books are constant across every simulated season
    for _, m in played.iterrows():
        h, a = idx[m["home"]], idx[m["away"]]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        gf[:, h] += hg; ga[:, h] += ag
        gf[:, a] += ag; ga[:, a] += hg
        pts[:, h] += 3 if hg > ag else (1 if hg == ag else 0)
        pts[:, a] += 3 if ag > hg else (1 if hg == ag else 0)

    for _, m in remaining.iterrows():
        h, a = idx[m["home"]], idx[m["away"]]
        lh, la = model.lambdas(m["home"], m["away"])
        grid = scoreline_grid(lh, la, model.rho)
        flat = grid.ravel()
        draws = rng.choice(flat.size, size=n, p=flat)
        hg, ag = np.unravel_index(draws, grid.shape)
        hg = hg.astype(np.int32); ag = ag.astype(np.int32)

        gf[:, h] += hg; ga[:, h] += ag
        gf[:, a] += ag; ga[:, a] += hg
        pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

    gd = gf - ga
    # PL tie-breakers: points, then goal difference, then goals for. Packed into
    # one sortable key (gd is offset to stay non-negative; gf < 1000).
    key = pts.astype(np.int64) * 10**7 + (gd.astype(np.int64) + 500) * 10**3 + gf
    order = np.argsort(-key, axis=1, kind="stable")        # team indices, best first
    position = np.empty_like(order)
    rows = np.arange(n)[:, None]
    position[rows, order] = np.arange(T)[None, :]          # 0-based finishing place

    out = []
    for t, i in idx.items():
        place = position[:, i]
        out.append({
            "team": t,
            "proj_points": round(float(pts[:, i].mean()), 1),
            "title_pct": round(100 * float((place == 0).mean()), 1),
            "top4_pct": round(100 * float((place < lg.europe_spots).mean()), 1),
            "relegation_pct": round(
                100 * float((place >= T - lg.relegation_spots).mean()), 1),
        })
    return pd.DataFrame(out).sort_values("proj_points", ascending=False)
