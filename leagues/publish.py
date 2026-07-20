"""Orchestrator: fit -> sim -> props -> picks -> data/leagues/pl.json.

The ONLY module that knows the published JSON contract. Everything else returns
plain frames and dicts, which is what makes generalising to four leagues a loop
rather than a rewrite.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import config, dataset, fixtures, odds, picks, players, props, second_tier, sim
from leagues.model import (LeagueModel, promoted_priors, score_for_outcome,
                           top_scorelines)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "leagues"
PICKS_DIR = ROOT / "data-raw" / "leagues"
MATCHWEEKS_AHEAD = 1
# A pick is only FROZEN once kickoff is near. Locking early would freeze a model
# that cannot yet see late form, injuries or the closing market, and the frozen
# pick would then contradict the probabilities shown beside it. Until a fixture
# enters this window its pick is provisional and recomputed every run.
#
# 45 minutes. Deliberately AFTER the confirmed starting XI, which clubs publish about
# an hour before kickoff -- the whole point is that a late team change reaches the
# model before the pick is committed.
#
# TIED TO THE PUBLISH CADENCE. A pick can only freeze on a run landing inside this
# window, so the matchday workflow runs every 30 minutes: narrower than the window,
# giving each fixture roughly two chances to lock.
#
# THE COST, stated plainly: GitHub's scheduled runs are routinely delayed under load
# and are occasionally dropped. Lose both runs in a 45-minute window and the pick
# locks after kickoff, is marked tainted, and is VOIDED out of the record rather than
# backdated. That is the honest failure, but it is still a lost pick -- so if voids
# start appearing in the record, widen this window and the cadence TOGETHER.
LOCK_WINDOW_HOURS = 0.75
# A pick joins the high-confidence board at this probability. Chosen from a pooled
# walk-forward over all four leagues (3,958 matches), not guessed:
#     all picks   53.2%      p>=0.60  73.6%
#     p>=0.65     77.4%      p>=0.70  84.1%
# 0.65 trades a little hit rate for useful volume (~18 picks a matchweek across
# the four leagues, vs ~9 at 0.70). Membership is decided from the FROZEN
# probability at lock time -- never recomputed after a result, or winners could be
# selected in hindsight.
BEST_PICK_MIN_PROB = 0.65
# A team needs at least this many players in the rates table before it gets props
# at all. The sigma-lambda rescale makes a team's players sum to the match model's
# lambda, so with a near-empty squad it hands ONE man the entire team's expected
# goals: promoted Schalke had a single player with top-flight history and came out
# at a 72.8% anytime scorer, when the next-best number in all four leagues was
# 50.8%. A team with one player is more dangerous than a team with none, because
# none is visibly a hole and one looks like a great pick. Teams below this get no
# props, exactly like teams with no data at all.
MIN_SQUAD_FOR_PROPS = 6
# The bar for the cross-league player board, PER MARKET -- the markets have very
# different ceilings and a single number cannot serve all three.
#   shots/sot at 0.70 is the requested bar and is comfortably reachable.
#   goal CANNOT be: anytime scorer tops out around 50% for an elite striker in a
#   great matchup (the best in all four leagues today is 50.8%), because a team
#   only scores ~1.5 goals and one man takes a fraction of them. A 0.70 bar would
#   leave the goalscorer section permanently EMPTY, so it is set at the level that
#   selects a comparable top slice. Every card publishes its own probability, so
#   nothing here is presented as more certain than it is.
PLAYER_PICK_MIN_PROB = {"goal": 0.40, "shots": 0.70, "sot": 0.70}
PROP_FIELD = {"goal": "anytime_pct", "shots": "p_shots_2plus", "sot": "p_sot_1plus"}


def _confidence(p_pick: float) -> int:
    """1-5, matching the WC app's banding."""
    for threshold, conf in ((0.70, 5), (0.60, 4), (0.50, 3), (0.40, 2)):
        if p_pick >= threshold:
            return conf
    return 1


def _player_pick_publishable(hours_out: float, lineup_ready: bool) -> bool:
    """Provisional props may be explored early; locked props require both XIs."""
    return hours_out > LOCK_WINDOW_HOURS or lineup_ready


def _why(model, home: str, away: str) -> dict | None:
    """The drivers behind a pick, in plain multipliers.

    The model computes lambda_home = exp(attack[home] + defence[away] + home_adv),
    so those three terms ARE the reasoning -- they were simply never published. A
    card showing 78% and nothing else asks to be taken on faith; showing that
    Arsenal attack +34% and Coventry defend -28% lets a reader check the pick
    against what they know about the fixture, and spot a wrong one BEFORE the
    result does.

    Expressed as percentage deviations from the league average, because the raw
    log-scale coefficients mean nothing to anyone reading a football page. Note the
    sign convention: `defence` is positive when a side concedes MORE, so it is
    negated here to read the way people expect (higher = better defence).
    """
    try:
        if home not in model.attack or away not in model.attack:
            return None
        att = list(model.attack.values())
        dfc = list(model.defence.values())
        a_mean = sum(att) / len(att)
        d_mean = sum(dfc) / len(dfc)
        pc = lambda x: round(100.0 * (float(np.exp(x)) - 1.0), 1)
        return {
            "home_attack_pct": pc(model.attack[home] - a_mean),
            "home_defence_pct": pc(-(model.defence[home] - d_mean)),
            "away_attack_pct": pc(model.attack[away] - a_mean),
            "away_defence_pct": pc(-(model.defence[away] - d_mean)),
            "home_advantage_pct": pc(model.home_adv),
        }
    except Exception:
        return None            # explanation is a nicety; never fail a publish for it


def _market_block(mkt: dict | None, pred: dict, pick_type: str) -> dict | None:
    """Attach the de-vigged market line + the model's edge on its pick.

    `edge` = model probability minus market probability on the picked outcome:
    positive means the model rates the pick higher than the bookmakers do. It is a
    disagreement measure, NOT a claim of profit. None when no line is posted
    (off-season, or a fixture not yet priced)."""
    if not mkt:
        return None
    model_p = pred[f"p_{pick_type}"] if pick_type != "draw" else pred["p_draw"]
    return {**mkt, "edge": round(model_p - mkt[f"p_{pick_type}"], 3)}


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

    now = pd.Timestamp.now("UTC")           # utcnow() is deprecated in pandas 3+
    ref = now.tz_localize(None) if now.tzinfo else now
    squad_teams = sorted(set(fx["home"]) | set(fx["away"]))

    # Fit once to learn the strength scale, then refit with priors mapped onto it
    # -- promoted clubs have no top-flight history to fit on. Their prior comes from
    # their actual second-tier season (calibrated), NOT from ClubElo: ClubElo is a
    # single third-party point of failure, and the second-tier feed is the same
    # source as everything else here.
    base = LeagueModel().fit(matches, ref=ref)
    warnings = []
    no_history = [t for t in squad_teams if t not in base.attack]
    priors = {}
    if no_history:
        try:
            priors = second_tier.second_tier_priors(base, league, no_history)
        except Exception as exc:
            print(f"WARNING: second-tier feed unavailable for {league} ({exc})")
        still_missing = [t for t in no_history if t not in priors]
        if still_missing:
            # A promoted club we could not resolve in the second-tier feed (e.g.
            # promoted from a lower level, or an unmapped spelling) -> the honest
            # weakest-side fallback, and say so.
            priors.update(promoted_priors(base, still_missing))
            names = (", ".join(still_missing[:-1]) + f" and {still_missing[-1]}"
                     if len(still_missing) > 1 else still_missing[0])
            warnings.append(
                f"No second-tier record found for {names}, so they are seeded at "
                f"the strength of the league's weakest sides rather than by their "
                f"own form. Their projected finish is a rough placeholder.")
            print(f"WARNING: {league}: no second-tier prior for {still_missing}; "
                  f"weakest-side fallback")
    model = LeagueModel().fit(matches, ref=ref, priors=priors)

    # Squad freshness: during an open transfer window the player-club mapping is
    # only as current as the last verified pass over transfers.json. Say so on the
    # page rather than quietly showing a player at a club he has left.
    stale_days = players.transfers_age_days()
    if stale_days is not None and stale_days > 7:
        warnings.append(
            f"Squad lists were last checked against transfer news {stale_days} days "
            f"ago and the window is still open, so a player may appear at a club he "
            f"has since left.")

    # Only players who actually appeared last season, with realistic minutes.
    # Otherwise five seasons of departed players share out the team's expected
    # goals and every real striker is crushed to a few percent.
    squad = players.current_squad(logs)
    league_matches = 2 * (lg.n_teams - 1)
    exp_minutes = players.expected_minutes(logs, matches_per_season=league_matches)
    playing_time = players.playing_time(logs, matches_per_season=league_matches)
    rates = props.player_rates(logs, ref=ref)
    rates = rates[rates["player"].isin(squad)]
    rates, roster_incomplete, roster_unmatched = players.reconcile_rates_to_roster(
        rates, league)
    roster_age = players.roster_snapshot_age_hours()
    if roster_incomplete:
        warnings.append(
            f"Player markets are withheld for {', '.join(roster_incomplete)} because "
            f"the free current-roster source is incomplete (<"
            f"{players.MIN_COMPLETE_ROSTER} listed players).")
    if roster_unmatched:
        warnings.append(
            f"{len(roster_unmatched)} historical players could not be matched "
            f"strictly to the current roster and were excluded from player markets.")
    takers = players.penalty_takers(logs[logs["player"].isin(squad)])
    news = players.load_news(league)   # injuries/suspensions, Best Picks fixtures
    # Shot events are the ONLY per-match player feed, so without them a league can
    # neither offer a real shots-on-target number (the ratio would be a league
    # average, i.e. an assumption dressed as a measurement) nor grade any player
    # pick afterwards. Both consequences are surfaced rather than hidden.
    shots_ok = players.shot_events_available(league)
    if not shots_ok:
        warnings.append(
            f"{lg.name} has no shot-level feed, so player picks here cannot be "
            f"graded against actual match lines and the shots-on-target market is "
            f"withheld entirely.")
    concede = ctx["concede_factor"]
    pens_rate = ctx["pens_per_team_match"]

    table = sim.simulate_season(model, played, remaining, league)

    log_path = PICKS_DIR / lg.key.lower() / "picks_log.json"
    log = picks.load_log(log_path)
    # fixturedownload MatchNumbers reset to 1..N every season, but picks_log
    # persists across seasons, so namespace each entry by the season to stop next
    # season's fixture #1 inheriting (and being graded against) this season's pick.
    season_tag = lg.fixture_slug.rsplit("-", 1)[-1]
    log_key = lambda mid: f"{season_tag}:{mid}"

    # Player picks get their OWN frozen log, graded separately from the match picks.
    pl_log_path = PICKS_DIR / lg.key.lower() / "player_picks_log.json"
    pl_log = picks.load_log(pl_log_path)

    # Live bookmaker lines for upcoming fixtures (empty off-season -> every
    # match gets market: None; the card renders fine either way).
    market_odds = odds.fetch_fixture_odds(league)

    if remaining.empty:
        upcoming = remaining
    else:
        # The current matchweek is the round of the SOONEST unplayed fixture, not
        # the lowest round number: a single postponed early-round game would
        # otherwise make min() return that stale round and hide the imminent week.
        next_round = int(remaining.sort_values("date").iloc[0]["round"])
        upcoming = remaining[remaining["round"] < next_round + MATCHWEEKS_AHEAD]

    # A squad too thin to share out the team's goals sensibly is dropped entirely
    # (see MIN_SQUAD_FOR_PROPS) rather than allowed to concentrate the whole team
    # lambda on one or two names.
    thin_squads = props.thin_squads(rates, squad_teams, MIN_SQUAD_FOR_PROPS)
    if thin_squads:
        rates = rates[~rates["team"].isin(thin_squads)]
    missing_squads = sorted({t for t in squad_teams
                             if rates[rates["team"] == t].empty})

    out_matches = []
    for _, m in upcoming.iterrows():
        home, away = m["home"], m["away"]
        pred = model.predict(home, away)
        probs = {home: pred["p_home"], "Draw": pred["p_draw"], away: pred["p_away"]}
        pick = max(probs, key=probs.get)

        # Freeze only inside the lock window; before that the pick stays live.
        hours_out = (pd.Timestamp(m["date"]) - now).total_seconds() / 3600.0
        if hours_out <= LOCK_WINDOW_HOURS:
            entry = picks.lock_pick(log, log_key(m["match_id"]), pick=pick,
                                    confidence=_confidence(probs[pick]),
                                    kickoff=m["date"], now=now,
                                    p_pick=probs[pick],
                                    board=probs[pick] >= BEST_PICK_MIN_PROB)
            provisional = False
        else:
            entry = {"pick": pick, "confidence": _confidence(probs[pick]),
                     "p_pick": round(float(probs[pick]), 4)}
            provisional = True
        # Everything the card shows must describe the FROZEN pick, not the fresh
        # argmax: on a re-run after the model flips, entry["pick"] is still the
        # locked side, so pick_type and the market edge below must be derived from
        # it -- otherwise the card shows one team but grades/edges another.
        frozen = entry["pick"]
        pick_type = ("home" if frozen == home
                     else "away" if frozen == away else "draw")
        # The model's committed single call. It is the most likely score GIVEN the
        # pick, so the card never contradicts itself -- the unconditional mode is
        # 1-1 in 68% of fixtures and would fight a home/away pick.
        score = score_for_outcome(pred["grid"], pick_type)
        # ...and the honest spread behind it. A single score is right ~12% of the
        # time; these three cover ~31%, and their probabilities show how thin the
        # call really is. `agrees_with_pick` marks which of them match the pick.
        spread = top_scorelines(pred["grid"], n=3)
        for s in spread:
            h, a = (int(x) for x in s["score"].split("-"))
            s["outcome"] = "home" if h > a else "away" if a > h else "draw"
            s["agrees_with_pick"] = (s["outcome"] == pick_type)

        # a player's shooting opportunity scales with how many shots his OPPONENT
        # concedes relative to the league average
        opp_factor = {home: concede.get(away, 1.0), away: concede.get(home, 1.0)}
        unavailable, doubtful = players.news_unavailable(news, (home, away))
        confirmed_starters, confirmed_bench = players.lineup_players(
            news, (home, away))
        lineup_ready = players.lineups_confirmed(news, (home, away))
        squad_props = props.match_props(
            rates, home, away, pred["lambda_home"], pred["lambda_away"],
            minutes=exp_minutes, pen_taker=takers, opp_shot_factor=opp_factor,
            exp_pens={home: pens_rate, away: pens_rate},
            unavailable=unavailable, doubtful=doubtful,
            playing_time=playing_time,
            confirmed_starters=confirmed_starters,
            confirmed_bench=confirmed_bench)

        # Player picks clearing their market's bar, frozen on the SAME schedule as
        # the match pick so both boards are graded under one discipline. A doubtful
        # player is deliberately still eligible: his halved expected minutes have
        # already pushed his probability down, so if he still clears the bar the
        # model is saying the pick survives the doubt.
        player_picks = []
        for market, field in PROP_FIELD.items():
            if market == "sot" and not shots_ok:
                continue          # synthetic ratio, not a measurement -- see above
            bar = PLAYER_PICK_MIN_PROB[market] * 100.0
            for p in squad_props:
                if p[field] < bar:
                    continue
                # Locked player picks require both confirmed XIs. A predicted XI
                # may be useful for a provisional board, but it is not evidence
                # strong enough to freeze a graded player proposition.
                if not _player_pick_publishable(hours_out, lineup_ready):
                    continue
                pkey = f"{log_key(m['match_id'])}:{market}:{p['player']}"
                prob = p[field] / 100.0
                if hours_out <= LOCK_WINDOW_HOURS:
                    pe = picks.lock_prop(pl_log, pkey, market=market,
                                         player=p["player"], team=p["team"],
                                         p_pick=prob, confidence=_confidence(prob),
                                         kickoff=m["date"], now=now,
                                         bar=PLAYER_PICK_MIN_PROB[market])
                    pprov = False
                else:
                    pe = {"p_pick": round(prob, 4), "confidence": _confidence(prob)}
                    pprov = True
                player_picks.append({
                    "market": market,
                    "line": picks.PROP_MARKETS[market][2],
                    "player": p["player"],
                    "team": p["team"],
                    "position": p["position"],
                    "p_pick": pe["p_pick"],
                    "confidence": pe["confidence"],
                    "provisional": pprov,
                    "doubt": p.get("doubt", False),
                    "penalty_taker": p.get("penalty_taker", False),
                    "appearance_pct": p.get("appearance_pct"),
                    "expected_minutes": p.get("expected_minutes"),
                    "lineup_confirmed": lineup_ready,
                    "gradeable": shots_ok,
                })
        player_picks.sort(key=lambda x: -x["p_pick"])

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
                "score": score,
                "top_scores": spread,
                "confidence": entry["confidence"],
                "provisional": provisional,   # True = not yet frozen; will be re-picked
                "p_pick": entry.get("p_pick"),
                "why": _why(model, home, away),
                "best_pick": bool((entry.get("p_pick") or 0) >= BEST_PICK_MIN_PROB),
                "reasons": [
                    f"Model: {home} {pred['p_home']:.0%} / draw {pred['p_draw']:.0%} "
                    f"/ {away} {pred['p_away']:.0%}",
                    f"Expected goals: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}",
                ],
            },
            "props": (props.top_props(squad_props, home)
                      + props.top_props(squad_props, away)),
            "player_picks": player_picks,
            "market": _market_block(odds.market_for(market_odds, home, away),
                                    pred, pick_type),
            "result": None,
            "graded": None,
            "void": False,
        })

    # Grade every played fixture we had locked a pick for, against the FROZEN pick.
    graded = []
    for _, m in played.iterrows():
        k = log_key(m["match_id"])
        entry = log.get(k)
        if not entry:
            continue
        g = picks.grade(entry, {"home": m["home"], "away": m["away"],
                                "home_goals": m["home_goals"],
                                "away_goals": m["away_goals"]})
        log[k].update({"graded": g["graded"], "void": g["void"]})
        graded.append(log[k])

    picks.save_log(log, log_path)
    picks.save_log(pl_log, pl_log_path)

    # Whole-season fixture list: every match, played or not, with its frozen pick
    # and grade. Deliberately WITHOUT props — 380 fixtures of scorer data would
    # bloat the payload; the props live on the current matchweek's cards only.
    season = []
    for _, m in fx.sort_values(["round", "date"]).iterrows():
        entry = log.get(log_key(m["match_id"])) or {}
        played_row = bool(m["played"])
        season.append({
            "id": int(m["match_id"]),
            "matchweek": int(m["round"]),
            "date": pd.Timestamp(m["date"]).isoformat(),
            "home": m["home"],
            "away": m["away"],
            "result": ({"home_goals": int(m["home_goals"]),
                        "away_goals": int(m["away_goals"])} if played_row else None),
            "pick": entry.get("pick"),
            "graded": entry.get("graded"),
            "void": bool(entry.get("void", False)),
        })

    def _read(path):
        p = PICKS_DIR / path
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # Props gate is stored per league (PL in props_report.json, others suffixed);
    # reading the bare PL file for every league published PL's props numbers on
    # La Liga / Bundesliga / Ligue 1 pages.
    props_file = ("props_report.json" if league == "PL"
                  else f"props_report_{league.lower()}.json")

    return {
        "league": lg.name,
        "updated": datetime.now(timezone.utc).isoformat(),
        "record": picks.record(graded),
        "matches": out_matches,
        "season": season,
        "table": table.to_dict(orient="records"),
        "backtest": _read("backtest_report.json").get(league, {}),
        "props_backtest": _read(props_file),
        "missing_squads": missing_squads,
        "thin_squads": thin_squads,
        "roster_incomplete": roster_incomplete,
        "roster_snapshot_age_hours": (
            None if roster_age is None else round(roster_age, 1)),
        "roster_unmatched_count": len(roster_unmatched),
        "data_warnings": warnings,
    }


FILE_FOR = {"PL": "pl.json", "LALIGA": "laliga.json",
            "BUNDESLIGA": "bundesliga.json", "LIGUE1": "ligue1.json"}


def _publish_one(league: str, fname: str) -> bool:
    """Build and atomically write one league. Returns True on success."""
    try:
        payload = build(league)
    except Exception as exc:              # one league's outage must not sink the rest
        print(f"ABORT {league}: {exc}; leaving its file untouched")
        return False
    path = OUT / fname
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)                     # atomic: never publish a half-written file
    print(f"wrote {path} - {len(payload['matches'])} fixtures, "
          f"{len(payload['table'])} teams")
    if payload.get("thin_squads"):
        print(f"  WARNING {league}: squad too thin for props "
              f"{payload['thin_squads']} (<{MIN_SQUAD_FOR_PROPS} players with "
              f"top-flight history) - no props, to stop one man absorbing the "
              f"whole team lambda")
    if payload.get("missing_squads"):
        print(f"  WARNING {league}: no player data for {payload['missing_squads']} "
              f"(promoted clubs have no top-flight history) - they get no props")
    return True


def build_best_picks() -> dict:
    """The high-confidence board: every league's strongest picks in one place.

    Assembled AFTER the leagues publish, by reading each league's frozen picks_log
    and its fixture results. Membership is decided by the probability recorded at
    LOCK time (>= BEST_PICK_MIN_PROB), so a pick cannot be promoted onto the board
    after it wins -- the same freezing discipline as the main record.

    Graded SEPARATELY from the all-picks record, so this tier can be judged on its
    own and shown to be earning its billing (or not).
    """
    upcoming, settled, incomplete = [], [], []
    seen_upcoming = set()          # (league_key, match id) already on the board
    for league, fname in FILE_FOR.items():
        lg = config.get(league)
        season_tag = lg.fixture_slug.rsplit("-", 1)[-1]
        log = picks.load_log(PICKS_DIR / lg.key.lower() / "picks_log.json")

        # UPCOMING comes from the freshly published payload, not the picks_log:
        # outside the 48h lock window a pick is deliberately provisional and has no
        # log entry yet, so reading only frozen picks would leave the board empty
        # all week. Those entries are marked provisional so the page can say the
        # pick may still move.
        try:
            payload = json.loads((OUT / fname).read_text(encoding="utf-8"))
        except Exception:
            payload = {"matches": []}
        for m in payload.get("matches", []):
            p = m.get("prediction", {})
            if not p.get("best_pick"):
                continue
            seen_upcoming.add((league, int(m["id"])))
            upcoming.append({
                "league": lg.name, "league_key": league,
                "id": m["id"], "matchweek": m["matchweek"], "date": m["date"],
                "home": m["home"], "away": m["away"],
                "pick": p["pick"], "confidence": p.get("confidence"),
                "p_pick": p.get("p_pick"), "score": p.get("score"),
                "provisional": bool(p.get("provisional")),
            })

        # SETTLED comes only from the frozen log -- graded honestly.
        #
        # A failure here must NOT be swallowed. `settled` and `record` are rebuilt
        # from scratch every run, so skipping a league silently deletes its entire
        # graded history from the published record -- and the deletion always
        # removes losses as readily as wins, i.e. it flatters the model on a five
        # second timeout. The board is refused entirely instead (see `incomplete`),
        # leaving the last good file in place.
        try:
            fx = fixtures.fetch_fixtures(league)
        except Exception as exc:
            print(f"  best-picks: CANNOT grade {league} ({exc}) -- refusing to "
                  f"publish a board that would drop its record")
            incomplete.append(lg.name)
            continue
        by_id = {int(r["match_id"]): r for _, r in fx.iterrows()}

        for key, entry in log.items():
            if not str(key).startswith(f"{season_tag}:"):
                continue                      # a previous season's entry
            # Membership was FROZEN at lock time. Only fall back to comparing
            # against the live constant for legacy entries written before `board`
            # existed -- never for new ones, or raising the bar would retroactively
            # delete settled picks from the record.
            on_board = entry.get("board")
            if on_board is None:
                on_board = (entry.get("p_pick") or 0) >= BEST_PICK_MIN_PROB
            if not on_board:
                continue
            mid = int(str(key).split(":", 1)[1])
            row = by_id.get(mid)
            if row is None:
                continue
            item = {
                "league": lg.name, "league_key": league,
                "id": mid, "matchweek": int(row["round"]),
                "date": pd.Timestamp(row["date"]).isoformat(),
                "home": row["home"], "away": row["away"],
                "pick": entry["pick"], "confidence": entry.get("confidence"),
                "p_pick": entry.get("p_pick"),
            }
            if bool(row["played"]):
                g = picks.grade(entry, {"home": row["home"], "away": row["away"],
                                        "home_goals": row["home_goals"],
                                        "away_goals": row["away_goals"]})
                item["result"] = {"home_goals": int(row["home_goals"]),
                                  "away_goals": int(row["away_goals"])}
                item["graded"] = g["graded"]
                item["void"] = g["void"]
                settled.append(item)
            elif (league, mid) not in seen_upcoming:
                # Inside the lock window BOTH sources describe the same fixture --
                # the payload copy (with scoreline and provisional flag) and this
                # one. Publishing both put a duplicate card, blank-scored, on the
                # board for every locked pick on matchday.
                upcoming.append(item)

    upcoming.sort(key=lambda x: (-(x["p_pick"] or 0), x["date"]))
    settled.sort(key=lambda x: x["date"], reverse=True)
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "_incomplete": incomplete,     # non-empty -> caller must NOT publish
        "min_probability": BEST_PICK_MIN_PROB,
        "record": picks.record(settled),
        "upcoming": upcoming,
        "settled": settled[:60],
        # Backtested expectation for this tier, so the page can state what the
        # board is worth rather than implying certainty. Pooled walk-forward over
        # all four leagues, n=3958: all picks 53.2%, p>=0.65 77.4% (+/-3.8).
        "backtested_hit_rate_pct": 77.4,
        "backtested_all_picks_pct": 53.2,
    }


def append_history(best: dict, players: dict) -> list:
    """One row per publish: the record so far, so drift becomes visible.

    The Grades tab shows where the record STANDS. It cannot show which way it is
    moving, and the warning sign that matters is not a bad weekend -- a 65% pick
    loses one time in three, so three straight losses is normal -- but stated
    confidence drifting away from observed results over dozens of picks. Without a
    time series that is unanswerable, which makes "is it still working?" a matter
    of feel. This makes it a matter of record.

    Append-only, one row per day: re-running on the same day replaces that day's
    row rather than inflating the series.
    """
    path = PICKS_DIR / "record_history.json"
    try:
        hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        hist = []

    today = datetime.now(timezone.utc).date().isoformat()
    br, pr = best.get("record", {}), players.get("record", {})
    # Stated vs observed on the SETTLED best picks: the calibration question.
    settled = best.get("settled", [])
    graded = [s for s in settled if s.get("graded") in ("correct", "wrong")]
    stated = (sum(s.get("p_pick") or 0 for s in graded) / len(graded)) if graded else None
    actual = (sum(s.get("graded") == "correct" for s in graded) / len(graded)) if graded else None

    row = {
        "date": today,
        "best": {"correct": br.get("correct", 0), "wrong": br.get("wrong", 0),
                 "total": br.get("total", 0)},
        "players": {"correct": pr.get("correct", 0), "wrong": pr.get("wrong", 0),
                    "total": pr.get("total", 0)},
        "stated_pct": None if stated is None else round(100 * stated, 1),
        "actual_pct": None if actual is None else round(100 * actual, 1),
    }
    hist = [h for h in hist if h.get("date") != today] + [row]
    hist.sort(key=lambda h: h["date"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    tmp.replace(path)
    return hist


def build_player_picks() -> dict:
    """The cross-league player board: goalscorer, shot attempts, shots on target.

    Same discipline as build_best_picks -- upcoming is read from the freshly
    published payloads (so the board is populated outside the lock window, flagged
    provisional), settled comes ONLY from the frozen log and is graded against the
    player's actual match line.

    Graded per market as well as overall, because the three markets are not
    comparable: a 45% goalscorer pick and a 78% shots pick are both "high
    confidence" for their market, and pooling them would produce a headline number
    that describes neither.
    """
    upcoming, settled, ungradeable, incomplete = [], [], [], []
    for league, fname in FILE_FOR.items():
        lg = config.get(league)
        season_tag = lg.fixture_slug.rsplit("-", 1)[-1]
        log = picks.load_log(PICKS_DIR / lg.key.lower() / "player_picks_log.json")

        try:
            payload = json.loads((OUT / fname).read_text(encoding="utf-8"))
        except Exception:
            payload = {"matches": []}
        for m in payload.get("matches", []):
            for pp in m.get("player_picks", []) or []:
                upcoming.append({**pp, "league": lg.name, "league_key": league,
                                 "id": m["id"], "date": m["date"],
                                 "home": m["home"], "away": m["away"]})

        if not log:
            continue
        try:
            fx = fixtures.fetch_fixtures(league)
        except Exception as exc:
            print(f"  player-picks: CANNOT grade {league} ({exc}) -- refusing to "
                  f"publish a board that would drop its record")
            incomplete.append(lg.name)
            continue
        by_id = {int(r["match_id"]): r for _, r in fx.iterrows()}

        # Actual per-match player lines. Empty when shot events are unreadable
        # (upstream Bundesliga crash) -- then every pick in that league stays
        # PENDING rather than being graded wrong against missing data.
        try:
            actuals = players.match_player_stats(league)
        except Exception as exc:
            print(f"  player-picks: no actuals for {league} ({exc})")
            actuals = pd.DataFrame()
        have_actuals = not actuals.empty
        if not have_actuals:
            # Distinguish a PERMANENT missing feed from a transient one. Bundesliga
            # genuinely has no shot events (upstream crash) and its picks can never
            # be graded -- that is worth stating on the page. A one-off timeout on a
            # league that normally grades fine is a different thing entirely, and
            # treating it as permanent would silently delete that league's record
            # AND print a flatly false claim that it has no shot feed.
            try:
                permanent = not players.shot_events_available(league)
            except Exception:
                permanent = False
            if permanent:
                ungradeable.append(lg.name)
            else:
                print(f"  player-picks: {league} actuals unavailable but its feed "
                      f"normally works -- refusing to publish a partial record")
                incomplete.append(lg.name)
            continue
        if have_actuals:
            actuals = actuals.assign(day=pd.to_datetime(actuals["date"]).dt.date)
            idx = {(r["player"], r["day"]): r for _, r in actuals.iterrows()}

        for key, entry in log.items():
            parts = str(key).split(":")
            if len(parts) < 4 or parts[0] != season_tag:
                continue
            mid = int(parts[1])
            row = by_id.get(mid)
            if row is None:
                continue
            item = {**entry, "league": lg.name, "league_key": league, "id": mid,
                    "line": picks.PROP_MARKETS[entry["market"]][2],
                    "date": pd.Timestamp(row["date"]).isoformat(),
                    "home": row["home"], "away": row["away"]}
            if not bool(row["played"]):
                continue                  # already covered by the payload read above
            day = pd.Timestamp(row["date"]).date()
            actual = idx.get((entry["player"], day))
            settled.append(picks.grade_prop(entry, None if actual is None
                                            else dict(actual)) | item)

    upcoming.sort(key=lambda x: (-(x["p_pick"] or 0), x["date"]))
    settled.sort(key=lambda x: x["date"], reverse=True)
    by_market = {mk: picks.record([s for s in settled if s.get("market") == mk])
                 for mk in picks.PROP_MARKETS}
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "min_probability": PLAYER_PICK_MIN_PROB,
        "markets": {k: v[2] for k, v in picks.PROP_MARKETS.items()},
        "record": picks.record(settled),
        "record_by_market": by_market,
        "ungradeable_leagues": sorted(set(ungradeable)),
        "_incomplete": incomplete,     # non-empty -> caller must NOT publish
        "upcoming": upcoming,
        "settled": settled[:120],
    }


def main(argv=None):
    """Publish all four leagues, or just the ones named on the command line
    (e.g. `python -m leagues.publish PL` for quick iteration)."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    leagues = [a.upper() for a in argv] or list(FILE_FOR)
    OUT.mkdir(parents=True, exist_ok=True)
    attempted = ok = 0
    for league in leagues:
        if league not in FILE_FOR:
            print(f"skip {league!r}: unknown league; known {list(FILE_FOR)}")
            continue
        attempted += 1
        ok += _publish_one(league, FILE_FOR[league])
    # Cross-league high-confidence board, built from the frozen picks of every
    # league that just published.
    # Cross-league boards must represent one complete four-league refresh. If a
    # league failed, reading its previous output here would give stale picks a new
    # board timestamp and defeat the browser's staleness warning.
    full_refresh = set(leagues) == set(FILE_FOR) and attempted == len(FILE_FOR)
    boards_safe = full_refresh and ok == attempted
    if boards_safe:
        best = build_best_picks()
        bp = OUT / "best.json"
        if best["_incomplete"]:
            # Refuse rather than publish a record with a league's graded history
            # missing. The previous file stays -- stale by a run, but TRUE, which
            # is the right way round for a scoreboard.
            print(f"  SKIPPED best.json: could not grade {best['_incomplete']}; "
                  f"keeping the last complete board")
            bp = None
        if bp is not None:
            tmp = bp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
            tmp.replace(bp)
            r = best["record"]
            print(f"wrote {bp} - {len(best['upcoming'])} upcoming high-confidence "
                  f"picks, record {r['correct']}-{r['wrong']}")

        pp = build_player_picks()
        ppath = OUT / "player_picks.json"
        if pp["_incomplete"]:
            print(f"  SKIPPED player_picks.json: could not grade "
                  f"{pp['_incomplete']}; keeping the last complete board")
        else:
            tmp = ppath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(pp, indent=2, default=str), encoding="utf-8")
            tmp.replace(ppath)
            pr = pp["record"]
            counts = {mk: sum(1 for u in pp["upcoming"] if u["market"] == mk)
                      for mk in picks.PROP_MARKETS}
            print(f"wrote {ppath} - {len(pp['upcoming'])} upcoming player picks "
                  f"{counts}, record {pr['correct']}-{pr['wrong']}")

        # Record history -- only when BOTH boards are complete, or a refused board
        # would write a row understating the record and permanently distort the
        # series. A gap in the history is honest; a wrong point is not.
        if not best["_incomplete"] and not pp["_incomplete"]:
            hist = append_history(best, pp)
            hpath = OUT / "record_history.json"
            tmp = hpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
            tmp.replace(hpath)
            print(f"wrote {hpath} - {len(hist)} snapshots")

    elif ok:
        print("SKIPPED cross-league boards: they require a complete successful "
              "four-league refresh")

    # A scheduled full refresh is atomic at the deployment boundary: individual
    # files may have been written locally, but callers must not commit or deploy
    # them unless all four builds succeeded.
    if attempted and (ok == 0 or (full_refresh and ok != attempted)):
        raise RuntimeError(
            f"only {ok}/{attempted} league publish(es) succeeded; refusing deployment")


if __name__ == "__main__":
    main()
