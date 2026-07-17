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
              kickoff, now=None) -> dict:
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
