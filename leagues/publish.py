"""Orchestrator: fit -> sim -> props -> picks -> data/leagues/pl.json.

The ONLY module that knows the published JSON contract. Everything else returns
plain frames and dicts, which is what makes generalising to four leagues a loop
rather than a rewrite.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leagues import config, dataset, elo, fixtures, picks, players, props, sim
from leagues.model import LeagueModel, elo_priors, promoted_priors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "leagues"
PICKS_DIR = ROOT / "data-raw" / "leagues"
MATCHWEEKS_AHEAD = 1


def _confidence(p_pick: float) -> int:
    """1-5, matching the WC app's banding."""
    for threshold, conf in ((0.70, 5), (0.60, 4), (0.50, 3), (0.40, 2)):
        if p_pick >= threshold:
            return conf
    return 1


def build(league: str = "PL") -> dict:
    lg = config.get(league)
    matches = dataset.build_matches(league)
    fx = fixtures.fetch_fixtures(league)
    logs = players.fetch_player_logs(league)
    ctx = players.team_shot_context(league)

    played = fx[fx["played"]].copy()
    remaining = fx[~fx["played"]].copy()

    # Fit on history plus whatever of the new season has already been played.
    if not played.empty:
        add = played[["date", "home", "away", "home_goals", "away_goals"]].copy()
        add["date"] = pd.to_datetime(add["date"]).dt.tz_localize(None)
        matches = pd.concat([matches, add], ignore_index=True)

    now = pd.Timestamp.utcnow()
    ref = now.tz_localize(None) if now.tzinfo else now
    squad_teams = sorted(set(fx["home"]) | set(fx["away"]))

    # Fit once to learn the strength scale, then refit with priors mapped onto it
    # -- promoted clubs have no top-flight history to fit on.
    base = LeagueModel().fit(matches, ref=ref)
    warnings = []
    try:
        ratings = elo.elo_for_league(league, squad_teams)
        priors = elo_priors(ratings, base)
    except elo.ClubEloUnavailable as exc:
        # Do NOT let this pass silently: without a prior, a promoted club would be
        # fitted at league average and quietly kept out of the relegation places.
        no_history = [t for t in squad_teams if t not in base.attack]
        priors = promoted_priors(base, no_history)
        warnings.append(
            f"ClubElo unavailable ({exc.__class__.__name__}); promoted clubs "
            f"{no_history} seeded at the strength of the league's weakest sides "
            f"instead of their true rating from the division below.")
        print("WARNING:", warnings[-1])
    model = LeagueModel().fit(matches, ref=ref, priors=priors)

    # Only players who actually appeared last season, with realistic minutes.
    # Otherwise five seasons of departed players share out the team's expected
    # goals and every real striker is crushed to a few percent.
    squad = players.current_squad(logs)
    exp_minutes = players.expected_minutes(logs)
    rates = props.player_rates(logs, ref=ref)
    rates = rates[rates["player"].isin(squad)]
    takers = players.penalty_takers(logs[logs["player"].isin(squad)])
    concede = ctx["concede_factor"]
    pens_rate = ctx["pens_per_team_match"]

    table = sim.simulate_season(model, played, remaining, league)

    log_path = PICKS_DIR / lg.key.lower() / "picks_log.json"
    log = picks.load_log(log_path)

    if remaining.empty:
        upcoming = remaining
    else:
        next_round = int(remaining["round"].min())
        upcoming = remaining[remaining["round"] < next_round + MATCHWEEKS_AHEAD]

    missing_squads = sorted({t for t in squad_teams
                             if rates[rates["team"] == t].empty})

    out_matches = []
    for _, m in upcoming.iterrows():
        home, away = m["home"], m["away"]
        pred = model.predict(home, away)
        probs = {home: pred["p_home"], "Draw": pred["p_draw"], away: pred["p_away"]}
        pick = max(probs, key=probs.get)
        pick_type = "home" if pick == home else "away" if pick == away else "draw"

        entry = picks.lock_pick(log, m["match_id"], pick=pick,
                                confidence=_confidence(probs[pick]),
                                kickoff=m["date"], now=now)

        # a player's shooting opportunity scales with how many shots his OPPONENT
        # concedes relative to the league average
        opp_factor = {home: concede.get(away, 1.0), away: concede.get(home, 1.0)}
        squad_props = props.match_props(
            rates, home, away, pred["lambda_home"], pred["lambda_away"],
            minutes=exp_minutes, pen_taker=takers, opp_shot_factor=opp_factor,
            exp_pens={home: pens_rate, away: pens_rate})

        out_matches.append({
            "id": int(m["match_id"]),
            "matchweek": int(m["round"]),
            "date": pd.Timestamp(m["date"]).isoformat(),
            "venue": m["venue"],
            "home": home,
            "away": away,
            "prediction": {
                "p_home": round(pred["p_home"], 3),
                "p_draw": round(pred["p_draw"], 3),
                "p_away": round(pred["p_away"], 3),
                "pick": entry["pick"],           # the FROZEN pick, never a fresh one
                "pick_type": pick_type,
                "score": pred["score"],
                "confidence": entry["confidence"],
                "reasons": [
                    f"Model: {home} {pred['p_home']:.0%} / draw {pred['p_draw']:.0%} "
                    f"/ {away} {pred['p_away']:.0%}",
                    f"Expected goals: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}",
                ],
            },
            "props": (props.top_props(squad_props, home)
                      + props.top_props(squad_props, away)),
            "result": None,
            "graded": None,
            "void": False,
        })

    # Grade every played fixture we had locked a pick for, against the FROZEN pick.
    graded = []
    for _, m in played.iterrows():
        entry = log.get(str(m["match_id"]))
        if not entry:
            continue
        g = picks.grade(entry, {"home": m["home"], "away": m["away"],
                                "home_goals": m["home_goals"],
                                "away_goals": m["away_goals"]})
        log[str(m["match_id"])].update({"graded": g["graded"], "void": g["void"]})
        graded.append(log[str(m["match_id"])])

    picks.save_log(log, log_path)

    def _read(path):
        p = PICKS_DIR / path
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    return {
        "league": lg.name,
        "updated": datetime.now(timezone.utc).isoformat(),
        "record": picks.record(graded),
        "matches": out_matches,
        "table": table.to_dict(orient="records"),
        "backtest": _read("backtest_report.json").get(league, {}),
        "props_backtest": _read("props_report.json"),
        "missing_squads": missing_squads,
        "data_warnings": warnings,
    }


def main():
    payload = build("PL")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pl.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)                     # atomic: never publish a half-written file
    print(f"wrote {path} - {len(payload['matches'])} fixtures, "
          f"{len(payload['table'])} teams")
    if payload["missing_squads"]:
        print(f"WARNING: no player data for {payload['missing_squads']} "
              f"(promoted clubs have no top-flight history) - they get no props")


if __name__ == "__main__":
    main()
