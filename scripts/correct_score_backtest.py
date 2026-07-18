"""How good is the CORRECT SCORE prediction? (Premier League)

Walk-forward, strictly causal (train on date < cutoff), scoring the exact scoreline
rather than the 1X2 outcome. Compares four strategies plus two naive baselines, so
we find out what the model is actually worth instead of assuming.

  grid_mode  - argmax of the full Dixon-Coles scoreline grid (the true mode)
  pick_cond  - most likely score GIVEN the 1X2 pick (what the card shows today)
  always_11  - always predict 1-1
  most_common- always predict the single most common score in the training data

Also reports top-3 hit rate and the mean probability the model assigned to the
score that actually happened (a sharper measure than raw accuracy).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid, outcome_probs, score_for_outcome

ROOT = Path(__file__).resolve().parents[1]


def run(league="PL", xi=0.003, xg_weight=0.75, min_train=760, step_days=7):
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(train, ref=cutoff)
        except Exception:
            continue
        # most common exact score in the training window (a real baseline)
        tc = (train["home_goals"].astype(int).astype(str) + "-"
              + train["away_goals"].astype(int).astype(str)).value_counts()
        common = tc.index[0]

        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            grid = scoreline_grid(lh, la, model.rho)
            actual = f"{int(m['home_goals'])}-{int(m['away_goals'])}"

            h, a = np.unravel_index(np.argmax(grid), grid.shape)
            grid_mode = f"{int(h)}-{int(a)}"

            ph, pdw, pa = outcome_probs(grid)
            pick_type = ("home" if ph >= pdw and ph >= pa
                         else "draw" if pdw >= pa else "away")
            pick_cond = score_for_outcome(grid, pick_type)

            # top-3 most likely scorelines
            flat = grid.ravel()
            top3_idx = np.argsort(-flat)[:3]
            top3 = [f"{int(i)}-{int(j)}" for i, j in
                    (np.unravel_index(k, grid.shape) for k in top3_idx)]

            # probability the model gave to what actually happened
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            p_actual = float(grid[hg, ag]) if hg < grid.shape[0] and ag < grid.shape[1] else 0.0

            rows.append({
                "actual": actual,
                "grid_mode": grid_mode,
                "pick_cond": pick_cond,
                "always_11": "1-1",
                "most_common": common,
                "top3_hit": actual in top3,
                "p_actual": p_actual,
            })

    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("no matches scored")

    # Exact-hit rate is a SATURATED, noise-dominated metric: at ~12% on ~1.1k
    # matches one standard error is ~1pp, so the 0.5pp gap to the always-1-1
    # baseline is not distinguishable from noise, and no model change can be
    # detected through it. Log-loss over the FULL flattened grid is a local
    # proper scoring rule -- it depends only on the probability assigned to what
    # actually happened -- so it can see improvements in grid SHAPE that the
    # argmax never reveals. (RPS is wrong here: it needs a natural ordering and
    # the scoreline grid is 2-D with none.)
    eps = 1e-12
    logloss = float(-np.log(np.clip(r["p_actual"], eps, None)).mean())

    def _paired_ci(a, b, iters=2000, seed=7):
        """Bootstrap 95% CI on the PAIRED difference in hit rate (a - b)."""
        rng = np.random.default_rng(seed)
        d = (a.to_numpy().astype(float) - b.to_numpy().astype(float))
        idx = rng.integers(0, len(d), size=(iters, len(d)))
        boot = d[idx].mean(axis=1)
        return (round(100 * float(np.percentile(boot, 2.5)), 2),
                round(100 * float(np.percentile(boot, 97.5)), 2))

    hit_mode = r["grid_mode"] == r["actual"]
    hit_11 = r["always_11"] == r["actual"]
    lo, hi = _paired_ci(hit_mode, hit_11)

    out = {
        "league": league,
        "n": int(len(r)),
        # --- the honest headline: a proper scoring rule over the whole grid ---
        "grid_logloss_nats": round(logloss, 4),
        "mean_prob_on_true_score": round(float(r["p_actual"].mean()), 4),
        # --- coverage: what a top-N presentation actually buys ---
        "top1_pct": round(100 * hit_mode.mean(), 2),
        "top3_pct": round(100 * r["top3_hit"].mean(), 2),
        # --- baselines, with an honest uncertainty statement ---
        "always_1_1_pct": round(100 * hit_11.mean(), 2),
        "most_common_pct": round(100 * (r["most_common"] == r["actual"]).mean(), 2),
        "pick_conditional_pct": round(100 * (r["pick_cond"] == r["actual"]).mean(), 2),
        "mode_minus_baseline_pp": round(100 * (hit_mode.mean() - hit_11.mean()), 2),
        "mode_minus_baseline_95ci_pp": [lo, hi],
        "edge_is_significant": bool(lo > 0),
        "most_predicted_by_grid_mode": r["grid_mode"].value_counts().head(3).to_dict(),
        "actual_score_frequency": r["actual"].value_counts(normalize=True).head(5).round(4).to_dict(),
    }
    return out


def main():
    rep = run("PL")
    print(json.dumps(rep, indent=2))
    path = ROOT / "data-raw" / "leagues" / "correct_score_report.json"
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
