"""Honest pick tracking, ported from the World Cup engine.

Three rules, learned the hard way (the WC app had to void 14 picks):
  1. A pick is LOCKED before kickoff and never changed afterwards.
  2. The FROZEN pick is what gets graded -- never a hindsight re-pick.
  3. A pick first locked after kickoff is TAINTED -> void: shown, but excluded
     from the record.
"""
import json
from pathlib import Path

import pandas as pd

LATE_LOCK_HOURS = 2.5


def load_log(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_log(log: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def lock_pick(log: dict, match_id, pick: str, confidence: int,
              kickoff, now=None, p_pick: float | None = None) -> dict:
    """Record a pick, once. A second call for the same match is a no-op."""
    match_id = str(match_id)
    if match_id in log:
        return log[match_id]

    now = _utc(now if now is not None else pd.Timestamp.now("UTC"))
    kickoff = _utc(kickoff)
    late_by = (now - kickoff).total_seconds() / 3600.0

    log[match_id] = {
        "pick": pick,
        "confidence": int(confidence),
        # The probability AT LOCK TIME. Stored so the high-confidence selection is
        # frozen with the pick: deciding after the result which picks counted as
        # "best" would let winners be chosen in hindsight, which is exactly the
        # dishonesty the freezing exists to prevent.
        "p_pick": None if p_pick is None else round(float(p_pick), 4),
        "locked_at": now.isoformat(),
        "kickoff": kickoff.isoformat(),
        "tainted": bool(late_by > LATE_LOCK_HOURS),
    }
    return log[match_id]


def grade(entry: dict, result: dict) -> dict:
    """Grade the FROZEN pick against the authoritative result."""
    out = dict(entry)
    if entry.get("tainted"):
        out["void"] = True
        out["graded"] = "void"
        return out

    hg, ag = int(result["home_goals"]), int(result["away_goals"])
    winner = result["home"] if hg > ag else result["away"] if ag > hg else "Draw"
    out["void"] = False
    out["graded"] = "correct" if entry["pick"] == winner else "wrong"
    return out


def lock_prop(log: dict, key, market: str, player: str, team: str,
              p_pick: float, confidence: int, kickoff, now=None) -> dict:
    """Freeze one player pick. Same discipline as lock_pick: write once, never
    rewrite, and store the probability AT LOCK TIME so board membership cannot be
    decided in hindsight."""
    key = str(key)
    if key in log:
        return log[key]

    now = _utc(now if now is not None else pd.Timestamp.now("UTC"))
    kickoff = _utc(kickoff)
    log[key] = {
        "market": market,
        "player": player,
        "team": team,
        "p_pick": round(float(p_pick), 4),
        "confidence": int(confidence),
        "locked_at": now.isoformat(),
        "kickoff": kickoff.isoformat(),
        "tainted": bool((now - kickoff).total_seconds() / 3600.0 > LATE_LOCK_HOURS),
    }
    return log[key]


# What each player market needs to hit. Keep the LINE here, next to the grader,
# so the published card and the grade can never drift apart.
PROP_MARKETS = {
    "goal":  ("goals", 1, "Anytime goalscorer"),
    "shots": ("shots", 2, "2+ shot attempts"),
    "sot":   ("sot",   1, "1+ shot on target"),
}


def grade_prop(entry: dict, actual: dict | None) -> dict:
    """Grade a FROZEN player pick against that player's actual match line.

    `actual` is his row from players.match_player_stats, or None when he has no
    shot events in the fixture.

    None is graded WRONG, not void. It means one of two things -- he did not play,
    or he played and never had a shot -- and the shot feed cannot tell them apart.
    A bookmaker voids the first and settles the second as a loss. Given we cannot
    distinguish them, we take the HARSHER reading on purpose: expected minutes are
    part of what the model claims, so a man the model expected to play and start
    shooting who did neither is a miss. This can only ever understate the record,
    never inflate it, which is the direction an honest scoreboard should err in.
    """
    out = dict(entry)
    if entry.get("tainted"):
        out["void"] = True
        out["graded"] = "void"
        return out

    field, line, _ = PROP_MARKETS[entry["market"]]
    got = int((actual or {}).get(field, 0))
    out["void"] = False
    out["actual"] = None if actual is None else int(got)
    out["graded"] = "correct" if got >= line else "wrong"
    return out


def record(entries: list[dict]) -> dict:
    """Aggregate a record in the same shape the WC app publishes."""
    rec = {"correct": 0, "wrong": 0, "total": 0, "void": 0, "pending": 0,
           "by_confidence": {}}
    for e in entries:
        g = e.get("graded")
        if g == "void":
            rec["void"] += 1
            continue
        if g not in ("correct", "wrong"):
            rec["pending"] += 1
            continue
        rec["total"] += 1
        rec[g] += 1
        bucket = rec["by_confidence"].setdefault(str(e.get("confidence", 0)),
                                                 {"correct": 0, "total": 0})
        bucket["total"] += 1
        if g == "correct":
            bucket["correct"] += 1
    return rec
