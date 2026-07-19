"""Does OPPONENT-ADJUSTING the xG strengths fix the compression?

Diagnosis: our two strength channels are not like-for-like.

  Dixon-Coles goals : sd 0.295  -- opponent-adjusted (the likelihood knows who
                                   each team played)
  xG channel        : sd 0.253  -- a RAW per-team mean of xG, no opponent term

`_xg_strengths` computes log(mean_xGF / league_avg). A team that happened to face
weak opponents looks good; one that faced the top three looks bad. That both
compresses the spread and biases individual teams -- and we blend 75% toward it.

Total goals = lambda_home + lambda_away, so compressed strengths compress every
fixture's expected total toward the league mean. That is the plausible mechanism
behind a 1-1 modal score in 87% of fixtures and an Over/Under 2.5 call (57.7%)
that cannot beat the 58.8% base rate -- while the market can (log-loss 0.6706).

Fix under test: fit xG with the SAME structure as the goal model --
log E[xG] = attack_i + defence_j + home_advantage -- via a Poisson GLM, which
sklearn fits on continuous non-negative targets (quasi-Poisson). Then compare.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from leagues import dataset
from leagues.model import LeagueModel
from leagues.weights import decay_weights

ROOT = Path(__file__).resolve().parents[1]


def adjusted_xg_strengths(df, ref, xi=0.003):
    """Opponent-adjusted xG attack/defence: log E[xG] = att_i + def_j + home."""
    d = df.dropna(subset=["home_xg", "away_xg"])
    if d.empty:
        return {}, {}
    w = decay_weights(d["date"], ref=ref, xi=xi).to_numpy()
    teams = sorted(set(d["home"]) | set(d["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(d), len(teams)

    # one row per team-match: attack of the scorer, defence of the opponent, home flag
    X = np.zeros((2 * n, 2 * T + 1))
    y = np.empty(2 * n)
    sw = np.empty(2 * n)
    for k, (_, r) in enumerate(d.iterrows()):
        h, a = idx[r["home"]], idx[r["away"]]
        X[k, h] = 1; X[k, T + a] = 1; X[k, -1] = 1           # home attacking
        y[k] = max(float(r["home_xg"]), 1e-6); sw[k] = w[k]
        X[n + k, a] = 1; X[n + k, T + h] = 1                  # away attacking
        y[n + k] = max(float(r["away_xg"]), 1e-6); sw[n + k] = w[k]

    m = PoissonRegressor(alpha=1e-6, max_iter=500, fit_intercept=True)
    m.fit(X, y, sample_weight=sw)
    coef = m.coef_
    att = {t: float(coef[idx[t]]) for t in teams}
    dfn = {t: float(coef[T + idx[t]]) for t in teams}
    # centre both, matching how the raw channel is centred before blending
    am, dm = np.mean(list(att.values())), np.mean(list(dfn.values()))
    return ({t: v - am for t, v in att.items()},
            {t: v - dm for t, v in dfn.items()})


def main(league="PL"):
    df = dataset.build_matches(league)
    played = df.dropna(subset=["home_goals", "away_goals"])
    ref = pd.to_datetime(played["date"]).max()

    mod = LeagueModel()
    raw_att, raw_def = mod._xg_strengths(played, ref)
    adj_att, adj_def = adjusted_xg_strengths(played, ref)

    # the goal channel, for reference (this one IS opponent-adjusted)
    full = LeagueModel().fit(played, ref=ref)

    ra = np.array(list(raw_att.values()))
    aa = np.array(list(adj_att.values()))
    out = {
        "league": league,
        "raw_xg_attack_sd": round(float(ra.std()), 3),
        "adjusted_xg_attack_sd": round(float(aa.std()), 3),
        "fitted_model_attack_sd": round(float(np.std(list(full.attack.values()))), 3),
        "target_real_epl": "0.30-0.35",
    }
    # how much do the two xG channels disagree per team?
    common = sorted(set(raw_att) & set(adj_att))
    diff = np.array([adj_att[t] - raw_att[t] for t in common])
    out["mean_abs_team_shift"] = round(float(np.abs(diff).mean()), 3)
    out["biggest_shifts"] = {
        t: round(adj_att[t] - raw_att[t], 3)
        for t in sorted(common, key=lambda t: -abs(adj_att[t] - raw_att[t]))[:5]}
    print(json.dumps(out, indent=2))
    (ROOT / "data-raw" / "leagues" / "xg_adjust_experiment.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())


# RESULT (PL, walk-forward, 2026-07-19) -- NEGATIVE. Hypothesis rejected. Do not re-run.
#
#   attack sd:  raw xG 0.253   opponent-adjusted xG 0.250   (fitted model 0.196)
#   metrics:    raw  grid-LL 2.9686  1X2 RPS 0.1990  O/U 57.7%
#               adj  grid-LL 2.9700  1X2 RPS 0.1984  O/U 57.3%
#
# Opponent-adjusting the xG channel does NOT widen the strength spread, and does
# not improve prediction. In a league everyone plays everyone home and away, so
# schedule strength is near-identical across teams and the adjustment has little
# to bite on -- unlike international football or a partial season. It does shift
# individual teams ~0.116 (biggest movers: Norwich, Burnley, Sunderland, Leeds --
# clubs present in only SOME of the five seasons, who genuinely faced different
# opposition mixes), i.e. it corrects bias without changing variance. That bias
# correction did not translate into better forecasts.
#
# WIDER POINT, worth keeping: the framing that our fitted spread (0.196) is "too
# narrow" versus real EPL strengths (~0.30-0.35) is itself misleading. Fitted
# estimates SHOULD be shrunk relative to true strengths -- that is exactly what
# empirical-Bayes shrinkage is for, and scripts/tune_prior_strength.py already
# located the optimum by log-loss. Residual narrowness is correct behaviour for
# prediction, not a defect. The genuine defect (a shrinkage guard that could never
# fire) was found and fixed separately.
