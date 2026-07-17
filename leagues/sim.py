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


def order_teams(pts, gd, gf, h2h_pts, h2h_gd, h2h_gf, tiebreak: str):
    """Finishing order (best first) for every simulated season, as team indices.

    `tiebreak` selects the league's real rule for clubs level on points:
      "gd"  -> goal difference, then goals for   (PL, Bundesliga, Ligue 1)
      "h2h" -> among clubs level on points, a MINI-LEAGUE of only their meetings:
               head-to-head points, then h2h goal difference, then h2h goals for,
               then overall goal difference, then overall goals for   (La Liga)

    The h2h branch resolves each maximal run of clubs tied on points by their true
    mini-table (points each took off the OTHERS IN THE GROUP), which is exact for
    groups of any size -- unlike an adjacent-pairwise comparison, which mis-ranks
    even non-cyclic 3-club ties (A beat B, B beat C, A drew C: A tops the mini-table
    but a pairwise pass cannot lift A above C).
    """
    n, T = pts.shape
    # base order: points, then GD, then GF (packed into one sortable key)
    key = pts.astype(np.int64) * 10**7 + (gd.astype(np.int64) + 500) * 10**3 + gf
    order = np.argsort(-key, axis=1, kind="stable")
    if tiebreak != "h2h":
        return order

    # Re-rank each run of clubs level on points by their mini-league. Only the
    # (rare) tied runs do any work; most positions are already settled by points.
    for s in range(n):
        o = order[s]
        p0 = 0
        while p0 < T:
            p1 = p0 + 1
            while p1 < T and pts[s, o[p1]] == pts[s, o[p0]]:
                p1 += 1
            if p1 - p0 >= 2:               # a genuine tie on points
                group = list(o[p0:p1])
                def mini_key(t, grp=group, s=s):
                    hp = sum(int(h2h_pts[t, u, s]) for u in grp if u != t)
                    hgd = sum(int(h2h_gd[t, u, s]) for u in grp if u != t)
                    hgf = sum(int(h2h_gf[t, u, s]) for u in grp if u != t)
                    return (hp, hgd, hgf, int(gd[s, t]), int(gf[s, t]))
                order[s, p0:p1] = sorted(group, key=mini_key, reverse=True)
            p0 = p1
    return order


def _h2h_tables(fixtures_idx, samples, T: int, n: int):
    """h2h_* [T,T,n]: points, goal difference, and goals FOR that team i took off
    team j (summed over their meetings), per simulated season."""
    h2h_pts = np.zeros((T, T, n), dtype=np.int16)
    h2h_gd = np.zeros((T, T, n), dtype=np.int16)
    h2h_gf = np.zeros((T, T, n), dtype=np.int16)
    for (h, a), (hg, ag) in zip(fixtures_idx, samples):
        h2h_pts[h, a] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0)).astype(np.int16)
        h2h_pts[a, h] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0)).astype(np.int16)
        diff = (hg - ag).astype(np.int16)
        h2h_gd[h, a] += diff
        h2h_gd[a, h] -= diff
        h2h_gf[h, a] += hg.astype(np.int16)
        h2h_gf[a, h] += ag.astype(np.int16)
    return h2h_pts, h2h_gd, h2h_gf


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

    # Head-to-head leagues need each fixture's sampled scoreline kept, not just the
    # running totals, so tied clubs can be separated by their meetings.
    needs_h2h = getattr(lg, "tiebreak", "gd") == "h2h"
    fixtures_idx, samples = [], []

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
        if needs_h2h:
            fixtures_idx.append((h, a)); samples.append((hg, ag))

    if needs_h2h:
        for _, m in played.iterrows():      # real results count in the h2h too
            h, a = idx[m["home"]], idx[m["away"]]
            hg = np.full(n, int(m["home_goals"]), dtype=np.int32)
            ag = np.full(n, int(m["away_goals"]), dtype=np.int32)
            fixtures_idx.append((h, a)); samples.append((hg, ag))
        h2h_pts, h2h_gd, h2h_gf = _h2h_tables(fixtures_idx, samples, T, n)
    else:
        h2h_pts = h2h_gd = h2h_gf = None

    gd = gf - ga
    order = order_teams(pts, gd, gf, h2h_pts, h2h_gd, h2h_gf,
                        getattr(lg, "tiebreak", "gd"))
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
