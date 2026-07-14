"""The props gate: does the player model beat a dumb baseline?

WHY A SEASON HOLDOUT AND NOT A WALK-FORWARD: the design called for per-matchweek
walk-forward calibration, which needs per-match player logs. The only free source
of those (FBref) scrapes a headless Chrome per match page -- ~8 hours per league.
Understat's shot events give us shots and goals per match, but they contain only
players who took a shot, so we cannot tell who played and did not shoot. Without
that denominator an anytime-scorer calibration curve would be silently wrong
(every player in the sample took a shot, so they all look sharper than they are).

So we gate on what the data can honestly support: train on seasons up to 2024-25,
predict each player's 2025-26 non-penalty goals and shots GIVEN the minutes he
actually played, and score against a position-prior baseline that knows nothing
about the player. Minutes are given to both models, which isolates the thing we
are actually testing -- the per-90 rates -- from minutes projection.

If it does not beat the baseline, the shrinkage/decay constants in props.py are
wrong and the props do not ship.
"""
import json

import numpy as np
import pandas as pd

from leagues import players, props

TRAIN_THROUGH = "2425"
TEST_SEASON = "2526"
MIN_TEST_MINUTES = 450        # ~5 full matches: below this, a season total is noise


def run(league: str = "PL") -> dict:
    logs = players.fetch_player_logs(league)

    train = logs[logs["season"] <= TRAIN_THROUGH]
    test = logs[logs["season"] == TEST_SEASON]
    if train.empty or test.empty:
        raise SystemExit(f"need both train (<={TRAIN_THROUGH}) and test ({TEST_SEASON}) seasons")

    # Fit rates as of the start of the test season -- nothing from it leaks in.
    ref = players.season_end(TRAIN_THROUGH)
    rates = props.player_rates(train, ref=ref).set_index("player")

    rows = []
    for _, p in test.iterrows():
        if p["minutes"] < MIN_TEST_MINUTES or p["player"] not in rates.index:
            continue
        r = rates.loc[p["player"]]
        n90 = p["minutes"] / 90.0

        pri_goal = props.GOAL_PRIORS.get(p["pos"], props.GOAL_PRIORS["MF"])
        pri_shot = props.SHOT_PRIORS.get(p["pos"], props.SHOT_PRIORS["MF"])

        rows.append({
            "player": p["player"],
            "n90": n90,
            "goals_actual": float(p["np_goals"]),
            "goals_model": float(r["rate90"]) * n90,
            "goals_base": pri_goal * n90,
            "shots_actual": float(p["shots"]),
            "shots_model": float(r["shots90"]) * n90,
            "shots_base": pri_shot * n90,
        })

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("no players survived the join; check name consistency")

    def mae(a, b):
        return float((d[a] - d[b]).abs().mean())

    report = {
        "league": league,
        "n_players": int(len(d)),
        "goals_mae": round(mae("goals_model", "goals_actual"), 3),
        "goals_mae_baseline": round(mae("goals_base", "goals_actual"), 3),
        "shots_mae": round(mae("shots_model", "shots_actual"), 3),
        "shots_mae_baseline": round(mae("shots_base", "shots_actual"), 3),
        "goals_corr": round(float(d["goals_model"].corr(d["goals_actual"])), 3),
        "shots_corr": round(float(d["shots_model"].corr(d["shots_actual"])), 3),
    }
    report["goals_lift"] = round(
        100 * (1 - report["goals_mae"] / report["goals_mae_baseline"]), 1)
    report["shots_lift"] = round(
        100 * (1 - report["shots_mae"] / report["shots_mae_baseline"]), 1)
    report["passes"] = bool(report["goals_mae"] < report["goals_mae_baseline"]
                            and report["shots_mae"] < report["shots_mae_baseline"])

    # A crude calibration check we CAN make honestly: bucket players by predicted
    # goals and compare the bucket's predicted total against its actual total.
    d["bucket"] = pd.cut(d["goals_model"], [0, 1, 3, 6, 10, 100],
                         labels=["0-1", "1-3", "3-6", "6-10", "10+"])
    cal = (d.groupby("bucket", observed=True)
             .agg(n=("player", "size"),
                  predicted=("goals_model", "mean"),
                  actual=("goals_actual", "mean")))
    report["calibration"] = [
        {"bucket": str(i), "n": int(r["n"]),
         "predicted": round(float(r["predicted"]), 2),
         "actual": round(float(r["actual"]), 2)}
        for i, r in cal.iterrows()
    ]
    return report


def main():
    rep = run("PL")
    print(json.dumps(rep, indent=2))
    print(f"\nGoals : MAE {rep['goals_mae']} vs baseline {rep['goals_mae_baseline']} "
          f"({rep['goals_lift']:+.1f}% better)")
    print(f"Shots : MAE {rep['shots_mae']} vs baseline {rep['shots_mae_baseline']} "
          f"({rep['shots_lift']:+.1f}% better)")
    print("\nGATE:", "PASS" if rep["passes"] else "FAIL")


if __name__ == "__main__":
    main()


# TUNING NOTE (2026-07-14). A full sweep of K_NINETIES x SEASON_DECAY x
# W_REALIZED_HIGH over {5,7,10,14,20} x {0.6,0.7,0.85} x {0.4,0.6} moved goals MAE
# only between 1.301 and 1.396 -- about 1% around the default -- and did NOT shift
# the one real bias: the model over-predicts elite scorers (top bucket ~13.3
# predicted vs 10.5 actual). That bias is structural, not a shrinkage-strength
# problem: high scorers regress toward the mean season over season, and shrinking
# toward a POSITION prior cannot express that. Picking the nominally-best cell on a
# ~310-player holdout would be fitting noise, so the defaults stand.
#
# It also matters less than it looks: match_props rescales every player so that a
# team's lambdas sum to the match model's team lambda, so a proportional
# over-prediction across a squad is divided straight back out. What survives the
# rescale is the ORDERING within the squad, which the bias preserves.
