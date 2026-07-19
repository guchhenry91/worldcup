"""Output sanity checks over the PUBLISHED payloads.

The unit tests prove the code does what it was written to do. This proves the
NUMBERS THAT REACH THE PAGE make sense to a human -- the class of bug that only
shows up by looking at the rendered card (a pick contradicting its own scoreline,
a team eliminated but still favoured, a striker at a club he left).

Run before every publish/deploy:  python -m scripts.sanity_check
Exits non-zero if anything fails, so it can gate the ops jobs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {"pl": ("PL", 20, 3), "laliga": ("LALIGA", 20, 3),
           "bundesliga": ("BUNDESLIGA", 18, 2), "ligue1": ("LIGUE1", 18, 2)}

fails, warns = [], []


def fail(where, msg):
    fails.append(f"[FAIL] {where}: {msg}")


def warn(where, msg):
    warns.append(f"[warn] {where}: {msg}")


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def check_league(fn, key, n_teams, releg):
    d = load(f"data/leagues/{fn}.json")
    L = f"{key}"

    # ---- the payload identifies itself correctly -------------------------
    pb = d.get("props_backtest") or {}
    if pb and pb.get("league") not in (key, None):
        fail(L, f"props_backtest belongs to {pb.get('league')}, not this league")
    bt = d.get("backtest") or {}
    if not bt.get("rps"):
        warn(L, "no match backtest attached")

    # ---- matches ---------------------------------------------------------
    for m in d["matches"]:
        tag = f"{L} {m['home']} v {m['away']}"
        p = m["prediction"]
        s = p["p_home"] + p["p_draw"] + p["p_away"]
        if abs(s - 1) > 0.005:
            fail(tag, f"1X2 probs sum to {s:.4f}")
        if m["home"] == m["away"]:
            fail(tag, "team plays itself")

        # pick must BE the most likely outcome
        best = max((p["p_home"], "home"), (p["p_draw"], "draw"), (p["p_away"], "away"))[1]
        if p["pick_type"] != best:
            fail(tag, f"pick_type={p['pick_type']} but {best} is most likely")
        # pick name must match pick_type
        expect = {"home": m["home"], "away": m["away"], "draw": "Draw"}[p["pick_type"]]
        if p["pick"] != expect:
            fail(tag, f"pick '{p['pick']}' contradicts pick_type {p['pick_type']}")

        # scoreline must agree with the pick  <-- the bug the user caught
        h, a = (int(x) for x in p["score"].split("-"))
        implied = "home" if h > a else "away" if a > h else "draw"
        if implied != p["pick_type"]:
            fail(tag, f"score {p['score']} contradicts pick {p['pick']}")
        if h > 6 or a > 6:
            warn(tag, f"implausible scoreline {p['score']}")

        # top-3 scoreline spread: ranked, plausible, and containing the committed call
        ts = p.get("top_scores") or []
        if ts:
            pcts = [t["pct"] for t in ts]
            if pcts != sorted(pcts, reverse=True):
                fail(tag, f"top_scores not ranked by probability: {pcts}")
            if not all(0 < t["pct"] < 100 for t in ts):
                fail(tag, f"top_scores probability out of range: {pcts}")
            if len({t["score"] for t in ts}) != len(ts):
                fail(tag, "duplicate scoreline in top_scores")
            for t in ts:
                h2, a2 = (int(x) for x in t["score"].split("-"))
                imp = "home" if h2 > a2 else "away" if a2 > h2 else "draw"
                if t.get("outcome") != imp:
                    fail(tag, f"top_score {t['score']} labelled {t.get('outcome')}, implies {imp}")
                if t.get("agrees_with_pick") != (imp == p["pick_type"]):
                    fail(tag, f"top_score {t['score']} agrees_with_pick flag wrong")
            if sum(pcts) > 100:
                fail(tag, f"top_scores probabilities sum to {sum(pcts):.1f}%")

        # confidence must match the banding of the picked probability
        pp = {"home": p["p_home"], "draw": p["p_draw"], "away": p["p_away"]}[p["pick_type"]]
        exp_conf = next((c for t, c in ((.70, 5), (.60, 4), (.50, 3), (.40, 2)) if pp >= t), 1)
        if p["confidence"] != exp_conf:
            fail(tag, f"confidence {p['confidence']} but picked prob {pp:.3f} implies {exp_conf}")

        # props
        for x in m.get("props", []):
            if x["team"] not in (m["home"], m["away"]):
                fail(tag, f"prop for {x['player']} of unrelated team {x['team']}")
            if not (0 <= x["anytime_pct"] <= 100):
                fail(tag, f"{x['player']} anytime {x['anytime_pct']}")
            if x["exp_shots"] < 0 or x["exp_sot"] < 0:
                fail(tag, f"{x['player']} negative shots")
            if x["exp_sot"] > x["exp_shots"] + 1e-9:
                fail(tag, f"{x['player']} SOT {x['exp_sot']} > shots {x['exp_shots']}")
            if x["anytime_pct"] > 85:
                warn(tag, f"{x['player']} anytime {x['anytime_pct']}% looks too high")
        for team in (m["home"], m["away"]):
            names = [x["player"] for x in m.get("props", []) if x["team"] == team]
            if len(names) != len(set(names)):
                fail(tag, f"duplicate player in {team} props")
            if len(names) > 3:
                fail(tag, f"{team} has {len(names)} props (max 3)")

        mk = m.get("market")
        if mk:
            ms = mk["p_home"] + mk["p_draw"] + mk["p_away"]
            if abs(ms - 1) > 0.02:
                fail(tag, f"market probs sum to {ms:.3f}")

    # ---- projected table -------------------------------------------------
    t = d["table"]
    if len(t) != n_teams:
        fail(L, f"table has {len(t)} teams, expected {n_teams}")
    if len({r["team"] for r in t}) != len(t):
        fail(L, "duplicate team in table")
    if [r["proj_points"] for r in t] != sorted((r["proj_points"] for r in t), reverse=True):
        fail(L, "table not sorted by projected points")
    for r in t:
        if not (5 <= r["proj_points"] <= 3 * 2 * (n_teams - 1)):
            fail(L, f"{r['team']} proj_points {r['proj_points']} implausible")
        for k in ("title_pct", "top4_pct", "relegation_pct"):
            if not (0 <= r[k] <= 100):
                fail(L, f"{r['team']} {k}={r[k]}")
    for k, expect in (("title_pct", 100), ("relegation_pct", releg * 100)):
        tot = sum(r[k] for r in t)
        if abs(tot - expect) > 2:
            fail(L, f"{k} sums to {tot:.1f}, expected ~{expect}")

    # ---- whole-season fixture list ---------------------------------------
    season = d.get("season", [])
    expect_n = n_teams * (n_teams - 1)
    if len(season) != expect_n:
        fail(L, f"season has {len(season)} fixtures, expected {expect_n}")
    per_team = {}
    for m in season:
        per_team[m["home"]] = per_team.get(m["home"], 0) + 1
        per_team[m["away"]] = per_team.get(m["away"], 0) + 1
    for team, n in per_team.items():
        if n != 2 * (n_teams - 1):
            fail(L, f"{team} plays {n} games, expected {2 * (n_teams - 1)}")
    mws = sorted({m["matchweek"] for m in season})
    if mws != list(range(1, 2 * (n_teams - 1) + 1)):
        fail(L, f"matchweeks not contiguous 1..{2 * (n_teams - 1)}")

    # ---- teams/colours ---------------------------------------------------
    cfile = {"pl": "clubs.json"}.get(fn, f"clubs_{fn}.json")
    clubs = load(f"data/leagues/{cfile}")
    for r in t:
        if r["team"] not in clubs:
            fail(L, f"no colour entry for {r['team']}")

    # ---- encoding --------------------------------------------------------
    blob = json.dumps(d, ensure_ascii=False)
    for bad in ("�", "Ã©", "Ã¡", "â€™"):
        if bad in blob:
            fail(L, f"mojibake {bad!r} in payload (encoding bug)")


def check_best_picks():
    """The high-confidence board must be honest: every entry above the stated bar,
    ranked, and its record consistent."""
    p = ROOT / "data" / "leagues" / "best.json"
    if not p.exists():
        warn("best", "best.json not published yet")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    thr = d.get("min_probability")
    if not thr:
        fail("best", "no min_probability stated")
        return
    up = d.get("upcoming", [])
    for x in up:
        if (x.get("p_pick") or 0) < thr:
            fail("best", f"{x['home']} v {x['away']} at {x.get('p_pick')} is below the "
                         f"stated {thr} bar")
        if x["pick"] not in (x["home"], x["away"], "Draw"):
            fail("best", f"pick {x['pick']!r} is not a participant in "
                         f"{x['home']} v {x['away']}")
    probs = [x.get("p_pick") or 0 for x in up]
    if probs != sorted(probs, reverse=True):
        fail("best", "upcoming picks are not ranked by confidence")
    for x in d.get("settled", []):
        if (x.get("p_pick") or 0) < thr:
            fail("best", f"settled entry {x['home']} v {x['away']} below the bar "
                         f"-- selection must be frozen, not recomputed")
    rec = d.get("record", {})
    if rec.get("total") and rec["correct"] + rec["wrong"] != rec["total"]:
        fail("best", "record correct+wrong != total")

    # Team-news freshness: a Best Pick kicking off soon must have had both clubs
    # news-checked recently. These are the picks carrying the 77.4% billing, so
    # publishing one featuring a striker confirmed out is the specific failure this
    # gate exists to prevent.
    import datetime as dt
    from leagues import players as _players
    now = dt.datetime.now(dt.timezone.utc)
    for x in up:
        try:
            ko = dt.datetime.fromisoformat(str(x["date"]))
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=dt.timezone.utc)
        except Exception:
            continue
        hours_out = (ko - now).total_seconds() / 3600.0
        if not (0 < hours_out <= 24):
            continue                       # only gate the imminent ones
        news = _players.load_news(x["league_key"])
        age = _players.news_checked_age_hours(news, (x["home"], x["away"]))
        if age is None:
            fail("best", f"{x['home']} v {x['away']} kicks off in {hours_out:.0f}h "
                         f"and has NOT been news-checked")
        elif age > 48:
            fail("best", f"{x['home']} v {x['away']} kicks off in {hours_out:.0f}h; "
                         f"team news is {age:.0f}h old")


def check_player_picks():
    """The player board, held to the same standard as the match board: every entry
    above its market's stated bar, ranked, graded consistently, and -- the one that
    actually bit -- never sourced from a squad too thin to share out the goals."""
    p = ROOT / "data" / "leagues" / "player_picks.json"
    if not p.exists():
        warn("players", "player_picks.json not published yet")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    bars = d.get("min_probability") or {}
    if not bars:
        fail("players", "no min_probability stated")
        return
    up = d.get("upcoming", [])
    for x in up:
        mk = x.get("market")
        if mk not in bars:
            fail("players", f"{x.get('player')} in unknown market {mk!r}")
            continue
        if (x.get("p_pick") or 0) < bars[mk]:
            fail("players", f"{x['player']} ({mk}) at {x.get('p_pick')} is below the "
                            f"stated {bars[mk]} bar")
        if x.get("team") not in (x.get("home"), x.get("away")):
            fail("players", f"{x['player']} plays for {x.get('team')}, not in "
                            f"{x.get('home')} v {x.get('away')}")
    probs = [x.get("p_pick") or 0 for x in up]
    if probs != sorted(probs, reverse=True):
        fail("players", "upcoming player picks are not ranked by confidence")

    # A shots-on-target pick may only come from a league with a real shot feed --
    # without one the on-target ratio is a league average, i.e. an assumption, and
    # publishing it as a 70%+ pick would dress a guess as a measurement.
    for x in up:
        if x.get("market") == "sot" and x.get("gradeable") is False:
            fail("players", f"SOT pick for {x['player']} in {x.get('league')}, which "
                            f"has no shot-level feed")

    # Every published prop must come from a squad the model could actually share
    # goals across. This is the Schalke guard: one player with top-flight history
    # absorbed his whole team's lambda and surfaced as a 72.8% scorer.
    for fn in ("pl.json", "laliga.json", "bundesliga.json", "ligue1.json"):
        f = ROOT / "data" / "leagues" / fn
        if not f.exists():
            continue
        lg = json.loads(f.read_text(encoding="utf-8"))
        thin = set(lg.get("thin_squads") or [])
        for m in lg.get("matches", []):
            for pr in (m.get("props") or []):
                if pr["team"] in thin:
                    fail("players", f"{fn}: prop published for {pr['team']}, whose "
                                    f"squad was flagged too thin")
            for pk in (m.get("player_picks") or []):
                if pk["team"] in thin:
                    fail("players", f"{fn}: player pick for {pk['team']}, whose "
                                    f"squad was flagged too thin")

    for x in d.get("settled", []):
        mk = x.get("market")
        if mk in bars and (x.get("p_pick") or 0) < bars[mk]:
            fail("players", f"settled entry {x.get('player')} below the bar -- "
                            f"selection must be frozen, not recomputed")
    rec = d.get("record", {})
    if rec.get("total") and rec["correct"] + rec["wrong"] != rec["total"]:
        fail("players", "record correct+wrong != total")
    tot = sum((d.get("record_by_market", {}).get(mk, {}) or {}).get("total", 0)
              for mk in bars)
    if tot != rec.get("total", 0):
        fail("players", f"per-market totals ({tot}) do not sum to the overall "
                        f"record ({rec.get('total', 0)})")


def check_squad_freshness():
    """Player-club attribution is only as fresh as data-raw/leagues/transfers.json.

    Understat has no in-progress-season data, so every displayed player is placed by
    LAST season's club until a verified transfer override says otherwise. This can't
    verify a squad by itself, but it can refuse to let the file go quietly stale
    while a window is open -- which is how a departed player keeps appearing.
    """
    import datetime as dt
    p = ROOT / "data-raw" / "leagues" / "transfers.json"
    if not p.exists():
        fail("transfers", "data-raw/leagues/transfers.json is missing entirely")
        return
    raw = json.loads(p.read_text(encoding="utf-8"))
    today = dt.date.today()
    # European summer window: mid-June to 1 September. Deals land daily.
    in_window = (today.month, today.day) >= (6, 10) and (today.month, today.day) <= (9, 2)
    if in_window:
        checked = raw.get("_verified_on")
        if not checked:
            warn("transfers", "window is OPEN and no _verified_on date recorded")
        else:
            try:
                age = (today - dt.date.fromisoformat(checked)).days
                # Deliberately a WARNING, not a failure. This runs unattended in CI:
                # blocking the whole publish would freeze results, tables and picks
                # too, which is worse than slightly stale rosters. publish.py instead
                # surfaces the staleness on the page as a data_warning, so readers are
                # told rather than the site going dark.
                if age > 7:
                    warn("transfers",
                         f"squads last verified {age} days ago ({checked}) with the "
                         f"window open -- re-verify (page shows a notice)")
                elif age > 3:
                    warn("transfers", f"squads last verified {age} days ago ({checked})")
            except ValueError:
                fail("transfers", f"_verified_on {checked!r} is not an ISO date")
        for lg in ("PL", "LALIGA", "BUNDESLIGA", "LIGUE1"):
            if lg not in raw:
                fail("transfers", f"no entry for {lg}")


def check_wc():
    p = ROOT / "data/predictions.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    k = d.get("knockout", {})
    bracket = k.get("bracket", [])
    odds = k.get("odds", [])

    # every team already knocked out must have 0% title odds
    eliminated = set()
    for m in bracket:
        r = m.get("result")
        if r and r.get("winner"):
            for side in ("home", "away"):
                if r.get(side) and r[side] != r["winner"]:
                    eliminated.add(r[side])
    for o in odds:
        if o["team"] in eliminated and o["champion"] > 0:
            fail("WC", f"{o['team']} is eliminated but has {o['champion']:.0%} title odds")

    # projected champion must be the top of the odds table
    if odds:
        top = max(odds, key=lambda o: o["champion"])["team"]
        if k.get("projected_champion") and k["projected_champion"] != top:
            fail("WC", f"projected champion {k['projected_champion']} != highest odds {top}")
        if k.get("projected_champion") in eliminated:
            fail("WC", f"projected champion {k['projected_champion']} is eliminated")

    # a finished tie must have a winner, and it must be one of the two teams
    for m in bracket:
        r = m.get("result")
        if m.get("status") == "final" and r:
            if not r.get("winner"):
                fail("WC", f"match {m['id']} final with no winner")
            elif r["winner"] not in (r.get("home"), r.get("away")):
                fail("WC", f"match {m['id']} winner {r['winner']} not a participant")

    rec = d.get("record", {})
    if rec.get("total") and rec["correct"] + rec["wrong"] != rec["total"]:
        fail("WC", "record correct+wrong != total")


def main():
    for fn, (key, n, releg) in LEAGUES.items():
        try:
            check_league(fn, key, n, releg)
        except FileNotFoundError:
            warn(key, "payload not published yet")
    check_squad_freshness()
    check_best_picks()
    check_player_picks()
    check_wc()

    for w in warns:
        print(w)
    for f in fails:
        print(f)
    print(f"\n{len(fails)} failure(s), {len(warns)} warning(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
