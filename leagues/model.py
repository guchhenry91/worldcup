"""The match model.

penaltyblog's DixonColesGoalModel requires INTEGER goals, so xG cannot be the
response. Instead we:
  1. fit Dixon-Coles on actual goals  -> rho, home advantage, attack/defence
  2. compute xG-based attack/defence separately (log ratio to league average)
  3. blend the two strength sets (default 75% xG / 25% goals)
  4. build the scoreline grid ourselves with the Dixon-Coles tau correction
This keeps penaltyblog's rigorous MLE while gaining xG's lower-variance signal.

penaltyblog's get_params() (verified empirically on real PL data) returns a
flat dict with keys "attack_{team}", "defence_{team}", "home_advantage" and
"rho". Sign convention: a STRONG defence has a MORE NEGATIVE value (e.g.
Arsenal -1.24 vs Burnley -0.63), which correctly lowers the opponent's
expected goals when added into the away lambda's log-rate.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

from leagues.weights import XI_PER_DAY, decay_weights

XG_WEIGHT = 0.75
MAX_GOALS = 10
PRIOR_STRENGTH = 6.0


def dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles low-score correction; negative rho lifts 0-0 and 1-1."""
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def scoreline_grid(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Normalized correct-score matrix [home, away]."""
    hp = poisson.pmf(np.arange(max_goals + 1), lh)
    ap = poisson.pmf(np.arange(max_goals + 1), la)
    grid = np.outer(hp, ap)
    for h in range(min(2, max_goals + 1)):
        for a in range(min(2, max_goals + 1)):
            grid[h, a] *= dc_tau(h, a, lh, la, rho)
    grid = np.clip(grid, 0.0, None)
    return grid / grid.sum()


def outcome_probs(grid: np.ndarray) -> tuple[float, float, float]:
    return (float(np.tril(grid, -1).sum()), float(np.trace(grid)),
            float(np.triu(grid, 1).sum()))


def elo_priors(elo: dict[str, float], model: "LeagueModel") -> dict[str, tuple[float, float]]:
    """Convert raw ClubElo ratings onto the model's LOG-STRENGTH scale.

    ClubElo is ~1500-2000; attack/defence live in a band roughly +/-0.5 wide.
    Feeding raw Elo into the strengths would blow lambda to the clip on every
    match. Instead map each club's Elo z-score onto the fitted spread of the
    league's strengths: a club one standard deviation above average in Elo gets
    a strength one standard deviation above average.
    """
    if not elo or not model.attack:
        return {}
    vals = np.array(list(elo.values()), dtype=float)
    mu = float(vals.mean())
    sd = float(vals.std()) or 1.0
    att = np.array(list(model.attack.values()), dtype=float)
    dfn = np.array(list(model.defence.values()), dtype=float)
    a_mu, a_sd = float(att.mean()), float(att.std())
    d_mu, d_sd = float(dfn.mean()), float(dfn.std())
    out = {}
    for team, rating in elo.items():
        z = (float(rating) - mu) / sd
        # stronger club: higher attack, more negative (better) defence
        out[team] = (a_mu + a_sd * z, d_mu - d_sd * z)
    return out


def promoted_priors(model: "LeagueModel", teams, n_weakest: int = 3) -> dict:
    """Fallback prior for clubs with no top-flight history, when ClubElo is down.

    A promoted side is NOT an average top-flight team -- that is the assumption
    that quietly keeps them out of the relegation places. Empirically they play
    like the division's weakest clubs, so seed them at the mean strength of the
    n weakest teams we HAVE fitted. Strictly worse than a real ClubElo prior
    (which knows how strong they were in the division below), so this is only a
    fallback -- but it is honest about the direction of the uncertainty.
    """
    if not model.attack:
        return {}
    # weakest = lowest attack + worst (least negative) defence
    net = {t: model.attack[t] - model.defence.get(t, 0.0) for t in model.attack}
    weakest = sorted(net, key=net.get)[:n_weakest]
    a = float(np.mean([model.attack[t] for t in weakest]))
    d = float(np.mean([model.defence[t] for t in weakest]))
    return {t: (a, d) for t in teams}


@dataclass
class LeagueModel:
    xi: float = XI_PER_DAY
    xg_weight: float = XG_WEIGHT
    rho: float = 0.0
    home_adv: float = 0.0
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)

    def fit(self, matches, ref=None, priors=None):
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        if df.empty:
            raise ValueError("no played matches to fit on")
        ref = ref or pd.to_datetime(df["date"]).max()
        w = decay_weights(df["date"], ref=ref, xi=self.xi).to_numpy().copy()

        clf = pb.models.DixonColesGoalModel(
            df["home_goals"].astype(int).to_numpy().copy(),
            df["away_goals"].astype(int).to_numpy().copy(),
            df["home"].to_numpy().copy(), df["away"].to_numpy().copy(), weights=w)
        clf.fit()
        goal_att, goal_def = self._parse_params(clf)

        xg_att, xg_def = self._xg_strengths(df, ref)
        teams = sorted(set(goal_att) | set(goal_def))

        # Blend DEVIATIONS, not raw values. penaltyblog identifies the model by
        # pinning mean(attack)=1 and letting defence absorb the league scoring
        # level (mean ~ -0.8), while the xG log-ratios are centred on 0. Mixing
        # the two raw scales shifts every lambda down ~12-18% — under-predicting
        # goals and inflating draws. Re-centring both on the goal model's level
        # keeps the league's true scoring rate and makes the xG channel purely a
        # (better) estimate of how far each team deviates from average.
        ga_mean = float(np.mean(list(goal_att.values()))) if goal_att else 0.0
        gd_mean = float(np.mean(list(goal_def.values()))) if goal_def else 0.0
        xa_mean = float(np.mean(list(xg_att.values()))) if xg_att else 0.0
        xd_mean = float(np.mean(list(xg_def.values()))) if xg_def else 0.0

        self.attack, self.defence = {}, {}
        for t in teams:
            ga, gd = goal_att.get(t, ga_mean), goal_def.get(t, gd_mean)
            xa, xd = xg_att.get(t), xg_def.get(t)
            if xa is None or xd is None:
                # no xG for this team: use its goal-based deviation only. Because
                # everything is expressed on the goal model's level, this team is
                # on the SAME scale as the rest (previously it became a superteam).
                self.attack[t], self.defence[t] = ga, gd
            else:
                w_xg = self.xg_weight
                self.attack[t] = ga_mean + w_xg * (xa - xa_mean) + (1 - w_xg) * (ga - ga_mean)
                self.defence[t] = gd_mean + w_xg * (xd - xd_mean) + (1 - w_xg) * (gd - gd_mean)

        self.level = (ga_mean, gd_mean)
        self._shrink_low_data(df, w, priors, ga_mean, gd_mean)
        return self

    def _parse_params(self, clf):
        """Extract attack/defence dicts + set self.rho/self.home_adv from
        penaltyblog's get_params().

        Verified empirically (Task 9, Step 3) on real PL data: get_params()
        returns a flat dict keyed "attack_{team}", "defence_{team}", plus
        scalar "home_advantage" and "rho" keys.
        """
        params = clf.get_params()
        self.rho = float(params["rho"])
        self.home_adv = float(params["home_advantage"])
        attack, defence = {}, {}
        for key, value in params.items():
            if key.startswith("attack_"):
                attack[key[len("attack_"):]] = float(value)
            elif key.startswith("defence_"):
                defence[key[len("defence_"):]] = float(value)
        return attack, defence

    def _xg_strengths(self, df, ref):
        if "home_xg" not in df or df["home_xg"].isna().all():
            return {}, {}
        d = df.dropna(subset=["home_xg", "away_xg"])
        if d.empty:
            return {}, {}
        wd = decay_weights(d["date"], ref=ref, xi=self.xi).to_numpy()
        rows = pd.concat([
            pd.DataFrame({"team": d["home"].values, "xgf": d["home_xg"].values,
                          "xga": d["away_xg"].values, "w": wd}),
            pd.DataFrame({"team": d["away"].values, "xgf": d["away_xg"].values,
                          "xga": d["home_xg"].values, "w": wd}),
        ])
        # No usable xG history (every match after ref, or a degenerate all-zero
        # window): skip the xG channel rather than divide by a zero weight-sum or
        # a zero league average (which would make log(f/avg) blow up to inf).
        if rows["w"].sum() <= 0:
            return {}, {}
        avg = np.average(rows["xgf"], weights=rows["w"])
        if avg <= 0:
            return {}, {}
        att, dfn = {}, {}
        for team, g in rows.groupby("team"):
            f = np.average(g["xgf"], weights=g["w"])
            a = np.average(g["xga"], weights=g["w"])
            att[team] = float(np.log(max(f, 0.05) / avg))
            dfn[team] = float(np.log(max(a, 0.05) / avg))
        return att, dfn

    def _shrink_low_data(self, df, w, priors, ga_mean, gd_mean):
        """Partial pooling: pull thinly-evidenced teams toward a prior.

        Without this, a promoted club fitted on one or two matches gets an absurd
        rating (a side that lost 0-4 on debut was rated 1.8% to win at home vs
        mid-table). Two prior sources, in order of preference:

        * `priors` — a team -> (attack, defence) mapping ALREADY on the log-strength
          scale (see `elo_priors`). Use for promoted clubs, where ClubElo knows the
          team from the division below and the league itself knows nothing.
        * otherwise the league average — plain hierarchical shrinkage toward the mean.

        A team with NO matches at all is seeded outright, so it becomes predictable
        instead of raising KeyError.
        """
        priors = priors or {}
        for team, p in priors.items():           # seed teams with zero history
            if team not in self.attack:
                self.attack[team], self.defence[team] = p

        for team in list(self.attack):
            mask = ((df["home"] == team) | (df["away"] == team)).to_numpy()
            eff = float(w[mask].sum()) if len(mask) == len(w) else 0.0
            k = PRIOR_STRENGTH / (PRIOR_STRENGTH + eff)      # k -> 1 when no data
            if k < 0.02:                                      # plenty of evidence
                continue
            pa, pdf = priors.get(team, (ga_mean, gd_mean))
            self.attack[team] = (1 - k) * self.attack[team] + k * pa
            self.defence[team] = (1 - k) * self.defence[team] + k * pdf

    def lambdas(self, home: str, away: str):
        for t in (home, away):
            if t not in self.attack:
                raise KeyError(f"team {t!r} not in fitted model")
        lh = np.exp(self.attack[home] + self.defence[away] + self.home_adv)
        la = np.exp(self.attack[away] + self.defence[home])
        return float(np.clip(lh, 0.05, 6.0)), float(np.clip(la, 0.05, 6.0))

    def predict(self, home: str, away: str) -> dict:
        lh, la = self.lambdas(home, away)
        grid = scoreline_grid(lh, la, self.rho)
        ph, pdw, pa = outcome_probs(grid)
        h, a = np.unravel_index(np.argmax(grid), grid.shape)
        return {"p_home": ph, "p_draw": pdw, "p_away": pa,
                "lambda_home": lh, "lambda_away": la,
                "score": f"{int(h)}-{int(a)}", "grid": grid}


def blend_probs(a, b, weight: float = 0.5):
    """Convex blend of two 1X2 forecasts, renormalized."""
    out = np.array(a, dtype=float) * weight + np.array(b, dtype=float) * (1 - weight)
    out = np.clip(out, 1e-9, None)
    out = out / out.sum()
    return float(out[0]), float(out[1]), float(out[2])


class Calibrator:
    """One-vs-rest isotonic recalibration of 1X2 probabilities, renormalized.
    Fit on a HELD-OUT period only — never on the training matches."""

    def __init__(self):
        self.iso = []

    def fit(self, probs: np.ndarray, outcomes: np.ndarray):
        self.iso = []
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(probs[:, k], (outcomes == k).astype(float))
            self.iso.append(ir)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.iso:
            return probs
        out = np.column_stack([self.iso[k].predict(probs[:, k]) for k in range(3)])
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)
