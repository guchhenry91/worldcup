"""How much does a team actually lose when a key attacker is absent?

The naive answer -- his whole share of the team's expected goals -- is wrong,
because someone replaces him. What matters is the gap between him and his
replacement, and that gap is what this measures.

METHOD. For every historical match, the fitted model gives an expected goals
figure that knows NOTHING about who played. The residual (actual - expected) is
therefore the part attributable to team news. Regressing that residual on the
share of the team's shooting that was missing estimates the marginal cost of an
absence, controlling for opponent and home advantage through the model itself.

ABSENCE PROXY, and its honest limitation: our only per-match player feed is shot
events, so "absent" means "took no shot". For a high-volume shooter that is a
decent proxy; for anyone else it is mostly noise, which is why the sample is
restricted to players averaging >=1.5 shots per 90. A quiet game still looks like
an absence, so any true effect here is ATTENUATED -- the measured slope is a lower
bound on the real one.

POSITION BUCKETS, and why DF/GK are not attempted here. run_by_position() splits
the same regression by attacking bucket (FW / AM / MF) to test whether they carry
genuinely different costs. Defenders and goalkeepers are deliberately EXCLUDED,
not just filtered out by the shots90 threshold: the absence proxy itself is "took
no shot", which is structurally uninformative for a position that legitimately
takes ~0 shots in a normal match. A missing center-back or keeper cannot be
detected this way, so there is no historical ground truth to regress against --
inventing a number here would not be a weaker estimate, it would be a guess wearing
a regression's clothes. See leagues/props.py absence_penalty() for how this
boundary is enforced in production.
"""
import numpy as np, pandas as pd
from leagues import dataset, players
from leagues.model import LeagueModel

MIN_SHOTS90 = 1.5          # below this, "took no shot" carries no information
ATTACKING_BUCKETS = ("FW", "AM", "MF")   # DF/GK excluded -- see module docstring


def _build_profile(league: str):
    """Shared setup for run() and run_by_position(): matches, per-match player
    events, the fitted model, and the shot-share profile used to detect
    absences. Returns None if the per-match player feed is unavailable."""
    matches = dataset.build_matches(league)
    matches["date"] = pd.to_datetime(matches["date"])
    ev = players.match_player_stats(league)
    if ev.empty:
        print(f"{league}: no per-match player feed; cannot measure")
        return None
    ev["day"] = pd.to_datetime(ev["date"]).dt.normalize()

    # per player: how often he shoots, and his share of his team's shots
    tot = ev.groupby(["team", "player"])["shots"].sum()
    games = ev.groupby(["team", "player"])["game_id"].nunique()
    team_games = ev.groupby("team")["game_id"].nunique()
    rate = (tot / games.clip(lower=1)).rename("shots_per_app")
    appearance = (games / team_games.reindex(games.index.get_level_values(0)).values)
    share = (tot / tot.groupby(level=0).sum()).rename("shot_share")
    prof = pd.concat([rate, share], axis=1)
    prof["appearance"] = appearance.values
    # regulars who shoot enough that silence is informative
    prof = prof[(prof["shots_per_app"] >= MIN_SHOTS90) & (prof["appearance"] >= 0.5)]

    model = LeagueModel().fit(matches, ref=matches["date"].max())
    played_days = ev.groupby("team")["day"].apply(set).to_dict()
    seen = ev.groupby(["team", "day"])["player"].apply(set).to_dict()

    # CONSECUTIVE-ABSENCE FILTER. "Took no shot" conflates two very different
    # things: a man who was injured, and a man who played badly. The second is a
    # SYMPTOM of the team performing poorly, so regressing goals on it measures
    # reverse causation and would make the model overreact to every absence.
    # A real absence (injury, suspension) spans a RUN of matches; a quiet game is
    # isolated. Requiring two consecutive misses keeps mostly the former.
    order = {t: sorted(days) for t, days in played_days.items()}
    genuine = {}          # (team, day) -> set of players absent for >=2 in a row
    for team, days in order.items():
        regs = prof.loc[team].index if team in prof.index.get_level_values(0) else []
        for i, day in enumerate(days):
            here = seen.get((team, day), set())
            prev = seen.get((team, days[i - 1]), set()) if i else set()
            nxt = seen.get((team, days[i + 1]), set()) if i + 1 < len(days) else set()
            run = {p for p in regs if p not in here
                   and (i and p not in prev or (i + 1 < len(days) and p not in nxt))}
            genuine[(team, day)] = run

    return matches, model, prof, played_days, genuine


def run(league="PL"):
    built = _build_profile(league)
    if built is None:
        return None
    matches, model, prof, played_days, genuine = built

    rows = []
    for _, m in matches.iterrows():
        day = m["date"].normalize()
        for side, opp, goals in (("home", "away", m["home_goals"]),
                                 ("away", "home", m["away_goals"])):
            team = m[side]
            if team not in model.attack or m[opp] not in model.attack:
                continue
            if day not in played_days.get(team, set()):
                continue                       # no shot data for this fixture
            lam = model.expected_goals(team, m[opp], home=(side == "home")) \
                  if hasattr(model, "expected_goals") else None
            if lam is None:
                a, d = model.attack[team], model.defence[m[opp]]
                lam = float(np.exp(a + d + (model.home_adv if side == "home" else 0.0)))
            regs = prof.loc[team] if team in prof.index.get_level_values(0) else None
            if regs is None or regs.empty:
                continue
            absent = genuine.get((team, day), set())
            missing = regs[regs.index.isin(absent)]
            rows.append({"missing_share": float(missing["shot_share"].sum()),
                         "resid": float(goals) - lam, "lam": lam})

    d = pd.DataFrame(rows)
    if len(d) < 200:
        print(f"{league}: only {len(d)} usable team-matches; too few")
        return None
    # slope of residual on missing share
    x, y = d["missing_share"].to_numpy(), d["resid"].to_numpy()
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    # bootstrap CI
    rng = np.random.default_rng(7)
    bs = [np.polyfit(x[i], y[i], 1)[0]
          for i in (rng.integers(0, n, n) for _ in range(400))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{league}: n={n}  mean missing share={x.mean():.3f}")
    print(f"   slope = {slope:+.3f} goals per unit of missing shot share "
          f"(95% CI {lo:+.3f} to {hi:+.3f})")
    print(f"   -> losing a player worth 25% of the team's shots costs "
          f"{-slope*0.25:+.3f} goals" if slope < 0 else
          f"   -> no cost detected")
    signif = hi < 0
    print(f"   statistically distinguishable from zero: {signif}")
    return {"league": league, "n": n, "slope": float(slope),
            "ci_low": float(lo), "ci_high": float(hi), "significant": bool(signif)}


def run_by_position(league="PL"):
    """Same regression as run(), but with the missing shot-share broken out by
    attacking position bucket (FW/AM/MF) so each can get its own cost -- IF the
    data supports telling them apart. Fits a multivariate OLS (one coefficient
    per bucket) instead of run()'s single pooled slope."""
    built = _build_profile(league)
    if built is None:
        return None
    matches, model, prof, played_days, genuine = built

    pos_logs = players.fetch_player_logs(league)
    pos_map = (pos_logs.groupby("player")["pos"]
              .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "MF"))
    prof = prof.copy()
    prof["pos"] = [pos_map.get(p, "MF") for p in prof.index.get_level_values(1)]
    # Not a filter step -- ATTACKING_BUCKETS already covers everyone who could
    # pass the shots90 threshold above in practice; this just makes the boundary
    # explicit and future-proof if that ever stops being true.
    prof = prof[prof["pos"].isin(ATTACKING_BUCKETS)]

    rows = []
    for _, m in matches.iterrows():
        day = m["date"].normalize()
        for side, opp, goals in (("home", "away", m["home_goals"]),
                                 ("away", "home", m["away_goals"])):
            team = m[side]
            if team not in model.attack or m[opp] not in model.attack:
                continue
            if day not in played_days.get(team, set()):
                continue
            lam = model.expected_goals(team, m[opp], home=(side == "home")) \
                  if hasattr(model, "expected_goals") else None
            if lam is None:
                a, d = model.attack[team], model.defence[m[opp]]
                lam = float(np.exp(a + d + (model.home_adv if side == "home" else 0.0)))
            regs = prof.loc[team] if team in prof.index.get_level_values(0) else None
            if regs is None or regs.empty:
                continue
            absent = genuine.get((team, day), set())
            missing = regs[regs.index.isin(absent)]
            row = {"resid": float(goals) - lam, "lam": lam}
            for bucket in ATTACKING_BUCKETS:
                b = missing[missing["pos"] == bucket]
                row[f"missing_{bucket}"] = float(b["shot_share"].sum())
            rows.append(row)

    d = pd.DataFrame(rows)
    if len(d) < 400:
        print(f"{league}: only {len(d)} usable team-matches; too few for a "
              f"{len(ATTACKING_BUCKETS)}-way split")
        return None

    cols = [f"missing_{b}" for b in ATTACKING_BUCKETS]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
    y = d["resid"].to_numpy()
    n = len(y)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)

    rng = np.random.default_rng(7)
    boots = []
    for _ in range(400):
        idx = rng.integers(0, n, n)
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boots.append(b)
    boots = np.array(boots)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)

    print(f"{league}: n={n}  multivariate slope by bucket "
          f"(pooled run() slope for comparison, not shown here)")
    out = {"league": league, "n": n, "intercept": float(coefs[0]), "buckets": {}}
    for i, bucket in enumerate(ATTACKING_BUCKETS, start=1):
        signif = hi[i] < 0
        print(f"   {bucket:>2}: slope {coefs[i]:+.3f}  "
              f"(95% CI {lo[i]:+.3f} to {hi[i]:+.3f})  "
              f"distinguishable from zero: {signif}")
        out["buckets"][bucket] = {"slope": float(coefs[i]), "ci_low": float(lo[i]),
                                  "ci_high": float(hi[i]), "significant": bool(signif)}
    return out


if __name__ == "__main__":
    import json, sys
    args = sys.argv[1:] or ["PL", "LALIGA", "LIGUE1"]
    pooled = [r for r in (run(l) for l in args) if r]
    json.dump(pooled, open("data-raw/leagues/absence_impact.json", "w"), indent=2)
    print()
    by_pos = [r for r in (run_by_position(l) for l in args) if r]
    json.dump(by_pos, open("data-raw/leagues/absence_impact_by_position.json", "w"),
              indent=2)
