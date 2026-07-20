"""World Cup 2026 prediction engine.

Reads data-raw/{schedule,ratings,news,results}.json and writes
data/predictions.json for the UI. Pure stdlib, deterministic.

Run: python predict.py
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone

import knockout

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:                       # no tzdata -> fixed EDT (WC is Jun-Jul)
    ET = timezone(timedelta(hours=-4))

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data-raw")
OUT = os.path.join(ROOT, "data")

HOST_COUNTRY = {"United States": "USA", "Mexico": "Mexico", "Canada": "Canada"}
HOME_ADV = 80          # Elo bonus for a host playing in its own country
INJURY_OUT = -15       # per key player out
INJURY_DOUBT = -8      # per key player doubtful
INJURY_CAP = -60
K_FACTOR = 60          # World Cup K for Elo updates
BASE_LAMBDA = 1.30     # avg goals per side at a neutral WC match

# city -> country, for host home advantage
US_CITIES = {"Atlanta", "Boston", "Foxborough", "Dallas", "Arlington", "Houston",
             "Kansas City", "Los Angeles", "Inglewood", "Miami", "Miami Gardens",
             "New York", "New Jersey", "East Rutherford", "Philadelphia",
             "San Francisco", "Santa Clara", "Seattle"}
MX_CITIES = {"Mexico City", "Guadalajara", "Zapopan", "Monterrey", "Guadalupe"}
CA_CITIES = {"Toronto", "Vancouver"}

FLAGS = {
    "Argentina": "🇦🇷", "France": "🇫🇷", "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Brazil": "🇧🇷",
    "Portugal": "🇵🇹", "Netherlands": "🇳🇱", "Belgium": "🇧🇪", "Germany": "🇩🇪", "Croatia": "🇭🇷",
    "Italy": "🇮🇹", "Morocco": "🇲🇦", "Uruguay": "🇺🇾", "Colombia": "🇨🇴", "United States": "🇺🇸",
    "Mexico": "🇲🇽", "Canada": "🇨🇦", "Japan": "🇯🇵", "South Korea": "🇰🇷", "Senegal": "🇸🇳",
    "Switzerland": "🇨🇭", "Denmark": "🇩🇰", "Austria": "🇦🇹", "Australia": "🇦🇺", "Ecuador": "🇪🇨",
    "Turkey": "🇹🇷", "Ukraine": "🇺🇦", "Poland": "🇵🇱", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Norway": "🇳🇴", "Sweden": "🇸🇪", "Czech Republic": "🇨🇿", "Czechia": "🇨🇿", "Greece": "🇬🇷",
    "Romania": "🇷🇴", "Serbia": "🇷🇸", "Slovakia": "🇸🇰", "Slovenia": "🇸🇮", "Hungary": "🇭🇺",
    "Albania": "🇦🇱", "Nigeria": "🇳🇬", "Egypt": "🇪🇬", "Algeria": "🇩🇿", "Tunisia": "🇹🇳",
    "Ivory Coast": "🇨🇮", "Ghana": "🇬🇭", "Cameroon": "🇨🇲", "South Africa": "🇿🇦", "Mali": "🇲🇱",
    "Burkina Faso": "🇧🇫", "Cape Verde": "🇨🇻", "DR Congo": "🇨🇩", "Iran": "🇮🇷", "Saudi Arabia": "🇸🇦",
    "Qatar": "🇶🇦", "Iraq": "🇮🇶", "Jordan": "🇯🇴", "UAE": "🇦🇪", "United Arab Emirates": "🇦🇪",
    "Uzbekistan": "🇺🇿", "China": "🇨🇳", "Indonesia": "🇮🇩", "Oman": "🇴🇲", "Bahrain": "🇧🇭",
    "Kuwait": "🇰🇼", "New Zealand": "🇳🇿", "Costa Rica": "🇨🇷", "Honduras": "🇭🇳", "Panama": "🇵🇦",
    "Jamaica": "🇯🇲", "Haiti": "🇭🇹", "Curacao": "🇨🇼", "Curaçao": "🇨🇼", "Trinidad and Tobago": "🇹🇹",
    "El Salvador": "🇸🇻", "Guatemala": "🇬🇹", "Suriname": "🇸🇷", "Peru": "🇵🇪", "Chile": "🇨🇱",
    "Paraguay": "🇵🇾", "Venezuela": "🇻🇪", "Bolivia": "🇧🇴", "Russia": "🇷🇺", "Israel": "🇮🇱",
    "North Macedonia": "🇲🇰", "Bosnia and Herzegovina": "🇧🇦", "Finland": "🇫🇮", "Ireland": "🇮🇪",
    "Republic of Ireland": "🇮🇪", "Northern Ireland": "🇬🇧", "Iceland": "🇮🇸", "Kosovo": "🇽🇰",
    "Montenegro": "🇲🇪", "Moldova": "🇲🇩", "Cyprus": "🇨🇾", "Estonia": "🇪🇪", "Latvia": "🇱🇻",
    "Lithuania": "🇱🇹", "Luxembourg": "🇱🇺", "Malta": "🇲🇹", "Armenia": "🇦🇲", "Georgia": "🇬🇪",
    "Azerbaijan": "🇦🇿", "Kazakhstan": "🇰🇿", "Belarus": "🇧🇾", "Angola": "🇦🇴", "Zambia": "🇿🇲",
    "Mozambique": "🇲🇿", "Benin": "🇧🇯", "Gabon": "🇬🇦", "Guinea": "🇬🇳", "Togo": "🇹🇬",
    "Madagascar": "🇲🇬", "Kenya": "🇰🇪", "Uganda": "🇺🇬", "Tanzania": "🇹🇿", "Libya": "🇱🇾",
    "Sudan": "🇸🇩", "Equatorial Guinea": "🇬🇶", "Namibia": "🇳🇦", "Mauritania": "🇲🇷",
    "Zimbabwe": "🇿🇼", "Sierra Leone": "🇸🇱", "Niger": "🇳🇪", "Gambia": "🇬🇲",
    "Guinea-Bissau": "🇬🇼", "Comoros": "🇰🇲", "Rwanda": "🇷🇼", "Botswana": "🇧🇼",
    "North Korea": "🇰🇵", "Syria": "🇸🇾", "Lebanon": "🇱🇧", "Palestine": "🇵🇸", "Vietnam": "🇻🇳",
    "Thailand": "🇹🇭", "Malaysia": "🇲🇾", "India": "🇮🇳", "Tajikistan": "🇹🇯",
    "Kyrgyzstan": "🇰🇬", "Turkmenistan": "🇹🇲", "Philippines": "🇵🇭", "Singapore": "🇸🇬",
    "Fiji": "🇫🇯", "New Caledonia": "🇳🇨", "Tahiti": "🇵🇫", "Solomon Islands": "🇸🇧",
    "Papua New Guinea": "🇵🇬", "Vanuatu": "🇻🇺",
}


def load(name, default=None):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def poisson(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def match_country(city):
    if city in MX_CITIES:
        return "Mexico"
    if city in CA_CITIES:
        return "Canada"
    return "United States"


def effective_elo(team, base_elo, news, reasons):
    """Apply injury adjustments from news; append human-readable reasons."""
    adj = 0
    info = (news or {}).get(team, {})
    key_names = {p.get("player") for p in info.get("key_players", [])}
    for inj in info.get("injuries", []):
        is_key = inj.get("player") in key_names
        if inj.get("status") == "out":
            adj += INJURY_OUT if is_key else int(INJURY_OUT / 2)   # trunc: -7, not floor -8
            reasons.append(f"{team}: {inj.get('player')} out ({inj.get('note', 'injury')})")
        elif inj.get("status") in ("doubtful", "suspended"):
            adj += INJURY_DOUBT if is_key else int(INJURY_DOUBT / 2)
            reasons.append(f"{team}: {inj.get('player')} {inj.get('status')}")
    adj = max(adj, INJURY_CAP)
    return base_elo + adj


ELO_PER_GOAL = 165.0   # ~1 goal of supremacy per 165 Elo
TOTAL_GOALS = 2.65     # expected combined goals in a WC group match
MAX_SUP = 3.0          # cap goal supremacy so no favorite is a lock
DRAW_PICK_MAX = 0.40   # if no team's win prob exceeds this, predict a draw
DC_RHO = -0.15         # Dixon-Coles low-score correction (raises draw prob;
                       # independent Poisson under-counts 0-0 & 1-1 draws)


def team_lambdas(dr):
    """Elo diff (incl. venue) -> (expected home goals, expected away goals)."""
    sup = max(-MAX_SUP, min(MAX_SUP, dr / ELO_PER_GOAL))
    return max(0.18, (TOTAL_GOALS + sup) / 2.0), max(0.18, (TOTAL_GOALS - sup) / 2.0)


def dc_tau(h, a, lh, la):
    """Dixon-Coles dependency factor for low scores — lifts the draw outcomes
    (0-0, 1-1) and trims 1-0/0-1, correcting the independent-Poisson draw bias."""
    if h == 0 and a == 0:
        return 1.0 - lh * la * DC_RHO
    if h == 0 and a == 1:
        return 1.0 + lh * DC_RHO
    if h == 1 and a == 0:
        return 1.0 + la * DC_RHO
    if h == 1 and a == 1:
        return 1.0 - DC_RHO
    return 1.0


def outcome_probs(dr):
    """Elo diff -> (p_home, p_draw, p_away, best_score) via a Dixon-Coles-
    corrected goal-supremacy Poisson.

    Calibrated so a ~200 Elo edge ~= a 55-60% favorite and even the biggest
    mismatches top out near ~88%, with realistic (~30%) draw rates.
    """
    lh, la = team_lambdas(dr)
    ph = pd = pa = 0.0
    best = {"home": (1, 0), "draw": (1, 1), "away": (0, 1)}
    bestp = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for h in range(9):
        for a in range(9):
            p = poisson(lh, h) * poisson(la, a) * dc_tau(h, a, lh, la)
            cat = "home" if h > a else "draw" if h == a else "away"
            if cat == "home":
                ph += p
            elif cat == "draw":
                pd += p
            else:
                pa += p
            if p > bestp[cat]:
                bestp[cat], best[cat] = p, (h, a)
    tot = ph + pd + pa
    scores = {k: f"{v[0]}-{v[1]}" for k, v in best.items()}
    return ph / tot, pd / tot, pa / tot, scores


def confidence(p):
    if p >= 0.65:
        return 5
    if p >= 0.54:
        return 4
    if p >= 0.47:
        return 3
    if p >= 0.40:
        return 2
    return 1


def knockout_kickoff(match):
    """Best available knockout kickoff.

    The bracket data currently carries dates but not times. Noon Eastern is a
    conservative fallback: it prevents an in-play model run from rewriting a pick.
    Adding ``time_et`` to bracket.json automatically makes this exact.
    """
    try:
        return datetime.strptime(
            f"{match['date']} {match.get('time_et') or '12:00'}",
            "%Y-%m-%d %H:%M",
        ).replace(tzinfo=ET)
    except (KeyError, TypeError, ValueError):
        return None


def freeze_knockout_pick(log, match, now=None):
    """Return the immutable pick for a knockout tie and store it when necessary."""
    now = now or datetime.now(ET)
    key = str(match["id"])
    existing = log.get(key)
    same_tie = (existing and existing.get("home") == match.get("home")
                and existing.get("away") == match.get("away"))
    kickoff = knockout_kickoff(match)
    started = kickoff is not None and now >= kickoff

    if same_tie:
        entry = existing
    else:
        entry = {
            "pick": match["pick"], "p_home": match["p_home"],
            "p_away": match["p_away"], "home": match["home"],
            "away": match["away"], "score": match["score"],
            "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tainted": bool(started),
        }
        log[key] = entry

    match["pick"] = entry["pick"]
    match["p_home"], match["p_away"] = entry["p_home"], entry["p_away"]
    match["score"] = entry.get("score", match["score"])
    match["void"] = bool(entry.get("tainted"))
    return entry


# --- player scorer model -------------------------------------------------
POS_RATE = {"FW": 0.55, "W": 0.40, "AM": 0.32, "MF": 0.18, "DF": 0.07}
BENCH_RESERVE = 1.45   # notional scoring weight of all non-listed players
ANYTIME_CAP = 0.62


def player_rate(p):
    """A player's baseline goals-per-match weight from club stats or position."""
    g, a = p.get("goals"), p.get("apps")
    if g is not None and a:
        rate = g / max(a, 1)
    else:
        rate = POS_RATE.get(p.get("pos", "FW"), 0.45)
    if p.get("pens"):
        rate *= 1.15
    return max(rate, 0.04)


def scorer_badge(prob, has_stats):
    """Confidence 1-3 + label for an anytime-scorer call."""
    if prob >= 0.42 or (prob >= 0.33 and has_stats):
        return 3, "High"
    if prob >= 0.20:
        return 2, "Medium"
    return 1, "Low"


def shots_tier(prob, pos):
    if pos in ("FW", "W") or prob >= 0.34:
        return "High"
    if pos in ("AM", "MF") or prob >= 0.20:
        return "Medium"
    return "Low"


def team_scorers(players, lam_team, injuries=None):
    """Top 3 likely scorers for a team given its expected goals this match.

    Players ruled out/suspended (per news injuries) are dropped; doubtful
    players are downweighted so a late fitness flag tempers their pick.
    """
    if not players:
        return []
    status = {}
    for inj in (injuries or []):
        nm = (inj.get("player") or "").lower()
        if nm:
            status[nm] = inj.get("status")
    rated = []
    for p in players:
        st = status.get((p.get("player") or "").lower())
        if st in ("out", "suspended"):
            continue
        r = player_rate(p)
        if st == "doubtful":
            r *= 0.55
        rated.append((p, r, st))
    if not rated:
        return []
    total = sum(r for _, r, _ in rated) + BENCH_RESERVE
    out = []
    for p, r, st in rated:
        share = r / total
        lam_p = lam_team * share
        prob = min(ANYTIME_CAP, 1 - math.exp(-lam_p))
        conf, label = scorer_badge(prob, p.get("goals") is not None)
        out.append({
            "player": p.get("player"),
            "club": p.get("club"),
            "pos": p.get("pos"),
            "goals": p.get("goals"),
            "apps": p.get("apps"),
            "pens": bool(p.get("pens")),
            "anytime": round(prob, 3),
            "shots": shots_tier(prob, p.get("pos", "")),
            "confidence": conf,
            "conf_label": label,
            "doubtful": st == "doubtful",
        })
    out.sort(key=lambda x: -x["anytime"])
    return out[:3]


def elo_update(elo_h, elo_w, gh, ga, dr):
    """eloratings.net update: returns delta for home team."""
    we = 1 / (1 + 10 ** (-dr / 400.0))
    w = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
    gd = abs(gh - ga)
    g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8.0
    return K_FACTOR * g * (w - we)


def main():
    schedule = load("schedule.json")
    ratings = load("ratings.json")
    news = (load("news.json", {}) or {}).get("teams", {})
    players = (load("players.json", {}) or {}).get("teams", {})
    results = load("results.json", {}) or {}  # {"1": {"home_goals":2,"away_goals":1}, ...}
    picks_log = load("picks_log.json", {}) or {}  # locked pre-match picks, id -> pred
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    elo = {t["team"]: float(t["elo"]) for t in ratings["teams"] if t.get("elo")}
    fifa = {t["team"]: t for t in ratings["teams"]}

    # Blend in FIFA points where Elo exists (85/15); FIFA points scaled ~ to Elo range
    base = {}
    for t in ratings["teams"]:
        name = t["team"]
        e = elo.get(name)
        fp = t.get("fifa_points")
        if e and fp:
            base[name] = 0.85 * e + 0.15 * (fp * 1.12)
        elif e:
            base[name] = e
        elif fp:
            base[name] = fp * 1.12
        else:
            base[name] = 1500.0

    # Apply Elo deltas from already-played results, in match order
    live = dict(base)
    played = []
    for m in sorted(schedule["matches"], key=lambda x: x["id"]):
        r = results.get(str(m["id"]))
        if not r:
            continue
        h, a = m["home"], m["away"]
        adv = HOME_ADV if match_country(m.get("city", "")) == h else 0
        dr = live[h] - live[a] + adv
        delta = elo_update(live[h], live[a], r["home_goals"], r["away_goals"], dr)
        live[h] += delta
        live[a] -= delta
        played.append(m["id"])

    # ...and from played knockout ties (ids 73+), so later rounds are rated on
    # current form. Knockout results carry their actual teams; pens count as
    # a draw for Elo (standard eloratings treatment of the 90/120-min score).
    bracket_json = load("bracket.json")
    ko_city = {}
    if bracket_json:
        for rd in bracket_json["rounds"]:
            for bm in rd["matches"]:
                ko_city[bm["id"]] = bm.get("city", "")
        if bracket_json.get("third_place"):
            tp = bracket_json["third_place"]
            ko_city[tp["id"]] = tp.get("city", "")
    for mid in sorted((int(k) for k in results if int(k) >= 73)):
        r = results[str(mid)]
        h, a = r.get("home"), r.get("away")
        if not h or not a or h not in live or a not in live:
            continue
        adv = HOME_ADV if match_country(ko_city.get(mid, "")) == h else 0
        dr = live[h] - live[a] + adv
        delta = elo_update(live[h], live[a], r["home_goals"], r["away_goals"], dr)
        live[h] += delta
        live[a] -= delta

    matches_out = []
    gmatches = []  # compact group-match list for the knockout simulator
    exp_pts = {}
    gd = {}        # team -> [goal difference, goals for] from real results
    for m in sorted(schedule["matches"], key=lambda x: (x["date"], x.get("time_et", ""), x["id"])):
        h, a = m["home"], m["away"]
        reasons = []
        eh = effective_elo(h, live.get(h, 1500), news, reasons)
        ea = effective_elo(a, live.get(a, 1500), news, reasons)
        adv = HOME_ADV if match_country(m.get("city", "")) == h else 0
        if adv:
            reasons.insert(0, f"{h} effectively at home in {m.get('city')} (+{HOME_ADV} Elo)")
        dr = eh - ea + adv
        ph, pd, pa, scores = outcome_probs(dr)
        lh, la = team_lambdas(dr)
        gmatches.append({"id": m["id"], "group": m["group"], "home": h, "away": a,
                         "lh": lh, "la": la, "city": m.get("city", ""),
                         "result": results.get(str(m["id"]))})
        scorers = {
            "home": team_scorers(players.get(h, []), lh, news.get(h, {}).get("injuries", [])),
            "away": team_scorers(players.get(a, []), la, news.get(a, {}).get("injuries", [])),
        }
        reasons.insert(0, f"Rating edge: {h} {eh:.0f} vs {a} {ea:.0f} ({dr:+.0f} incl. venue)")
        fh, fa = news.get(h, {}).get("form"), news.get(a, {}).get("form")
        if fh or fa:
            reasons.append(f"Form: {h} {fh or '—'} · {a} {fa or '—'}")

        # Pick the most likely team — but call a DRAW when the match is a
        # genuine toss-up (no side better than ~40% to win). The independent
        # Poisson keeps the draw from ever being the single highest outcome,
        # so without this the model is forced to guess a team in even games.
        fav_p = max(ph, pa)
        if fav_p <= DRAW_PICK_MAX and pd >= 0.23:
            pick, ptype, pmax = "Draw", "draw", pd
        elif ph >= pa:
            pick, ptype, pmax = h, "home", ph
        else:
            pick, ptype, pmax = a, "away", pa
        score = scores[ptype]  # displayed scoreline consistent with the pick
        cur_pred = {
            "p_home": round(ph, 3), "p_draw": round(pd, 3), "p_away": round(pa, 3),
            "pick": pick, "pick_type": ptype, "score": score,
            "confidence": confidence(pmax),
        }

        # --- pick tracking: lock each pick before kickoff, grade the locked
        # pick (not a hindsight re-computation) once the result is in. ---
        mid = str(m["id"])
        r = results.get(mid)
        graded = None
        try:
            kickoff = datetime.strptime(
                f"{m['date']} {m.get('time_et') or '12:00'}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        except ValueError:
            kickoff = None
        started = kickoff is not None and datetime.now(ET) >= kickoff
        if r is None:
            if not started or mid not in picks_log:
                # upcoming: keep the locked pick refreshed with the latest model
                entry = dict(cur_pred); entry["locked_at"] = now_iso
                picks_log[mid] = entry
            # in-play (kicked off, result pending): the lock stays frozen
            shown = picks_log[mid] if started else cur_pred
        else:
            # played: freeze and grade the pick locked before the result
            entry = picks_log.get(mid)
            if entry is None:  # match settled before tracking began — best effort
                entry = dict(cur_pred); entry["locked_at"] = now_iso
                entry["tainted"] = True    # cannot prove it pre-dates the result
                picks_log[mid] = entry
            # a pick "locked" >2.5h after kickoff may have seen the result
            # (hindsight) — grade it for display but void it from the record
            if "tainted" not in entry and kickoff is not None:
                try:
                    locked = datetime.fromisoformat(entry.get("locked_at", now_iso))
                    entry["tainted"] = locked > kickoff.astimezone(timezone.utc) + timedelta(hours=2.5)
                except ValueError:
                    entry["tainted"] = False
            actual = "home" if r["home_goals"] > r["away_goals"] else "away" if r["away_goals"] > r["home_goals"] else "draw"
            graded = "correct" if entry.get("pick_type") == actual else "wrong"
            shown = entry
        void = bool(r) and bool(picks_log.get(mid, {}).get("tainted"))

        exp_pts.setdefault(h, 0.0)
        exp_pts.setdefault(a, 0.0)
        if r:
            exp_pts[h] += 3 if r["home_goals"] > r["away_goals"] else 1 if r["home_goals"] == r["away_goals"] else 0
            exp_pts[a] += 3 if r["away_goals"] > r["home_goals"] else 1 if r["home_goals"] == r["away_goals"] else 0
            gd.setdefault(h, [0, 0]); gd.setdefault(a, [0, 0])
            gd[h][0] += r["home_goals"] - r["away_goals"]; gd[h][1] += r["home_goals"]
            gd[a][0] += r["away_goals"] - r["home_goals"]; gd[a][1] += r["away_goals"]
        else:
            exp_pts[h] += 3 * ph + pd
            exp_pts[a] += 3 * pa + pd

        matches_out.append({
            **{k: m.get(k) for k in ("id", "date", "time_et", "group", "home", "away", "venue", "city")},
            "status": "final" if r else "scheduled",
            "result": r,
            "graded": graded,
            "void": void,
            "prediction": {
                "p_home": shown["p_home"], "p_draw": shown["p_draw"], "p_away": shown["p_away"],
                "pick": shown["pick"], "pick_type": shown["pick_type"], "score": shown["score"],
                "confidence": shown["confidence"], "reasons": reasons[:4],
                "locked_at": shown.get("locked_at"),
            },
            "scorers": scorers,
        })

    standings = {}
    for g, teams in schedule["groups"].items():
        rows = [{"team": t, "exp_pts": round(exp_pts.get(t, 0.0), 1),
                 "gd": gd.get(t, [0, 0])[0], "gf": gd.get(t, [0, 0])[1],
                 "elo": round(live.get(t, 1500))} for t in teams]
        # FIFA tie-breakers: points, then goal difference, then goals scored
        rows.sort(key=lambda r: (-r["exp_pts"], -r["gd"], -r["gf"]))
        standings[g] = rows

    teams_out = {}
    for t in base:
        info = news.get(t, {})
        teams_out[t] = {
            "flag": FLAGS.get(t, "🏳️"),
            "elo": round(live.get(t, 1500)),
            "fifa_rank": fifa.get(t, {}).get("fifa_rank"),
            "form": info.get("form"),
            "form_note": info.get("form_note"),
            "injuries": info.get("injuries", []),
            "key_players": info.get("key_players", []),
            "news": info.get("news", []),
        }

    # only cleanly pre-match-locked picks count toward the official record;
    # picks locked after the fact ("void") are shown on cards but not scored
    graded = [m for m in matches_out if m["graded"] and not m["void"]]
    voided = sum(1 for m in matches_out if m["graded"] and m["void"])
    correct = sum(1 for m in graded if m["graded"] == "correct")
    by_conf = {}
    for m in graded:
        c = m["prediction"]["confidence"]
        slot = by_conf.setdefault(c, {"correct": 0, "total": 0})
        slot["total"] += 1
        slot["correct"] += 1 if m["graded"] == "correct" else 0
    record = {
        "correct": correct,
        "wrong": len(graded) - correct,
        "total": len(graded),
        "void": voided,
        "pending": sum(1 for m in matches_out if m["status"] != "final"),
        "by_confidence": by_conf,
    }

    knockout_out = None
    if bracket_json:
        sims = int(os.environ.get("WC_SIMS", "20000"))
        # knockout predictions should feel injuries too, not just raw Elo
        ko_elo = {t: effective_elo(t, live[t], news, []) for t in live}
        knockout_out = knockout.run(
            ko_elo, schedule["groups"], gmatches, standings, bracket_json, results,
            match_country, HOME_ADV, ELO_PER_GOAL, TOTAL_GOALS, MAX_SUP, sims=sims)

        # Lock knockout picks once and never rewrite them in play. A missing or
        # changed-team lock first seen after kickoff is retained for display but
        # marked void, because it cannot be proven to pre-date the match.
        for b in knockout_out["bracket"]:
            if not (b.get("home") and b.get("away")):
                continue
            freeze_knockout_pick(picks_log, b)

        # fold settled knockout ties into the overall record (pick vs advancer)
        ko_c = ko_t = ko_v = 0
        for b in knockout_out["bracket"]:
            if b.get("status") == "final" and b.get("winner") and b.get("void"):
                ko_v += 1
            elif b.get("status") == "final" and b.get("winner"):
                ko_t += 1
                if b["pick"] == b["winner"]:
                    ko_c += 1
        record["correct"] += ko_c
        record["wrong"] += ko_t - ko_c
        record["total"] += ko_t
        record["knockout"] = {"correct": ko_c, "total": ko_t, "void": ko_v}

    out = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record": record,
        "groups": schedule["groups"],
        "teams": teams_out,
        "matches": matches_out,
        "standings": standings,
        "knockout": knockout_out,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # persist the locked-pick log so the track record survives across runs
    with open(os.path.join(RAW, "picks_log.json"), "w", encoding="utf-8") as f:
        json.dump(picks_log, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(matches_out)} matches, {len(played)} played, "
          f"record {record['correct']}-{record['wrong']} of {record['total']} "
          f"({record['pending']} pending)")


if __name__ == "__main__":
    main()
