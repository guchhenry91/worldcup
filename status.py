"""Gate helper for the scheduled tasks. Prints what (if anything) needs doing:

{
  "finished_unrecorded": [ {id, home, away, date, time_et}, ... ],  # need a score
  "upcoming_4h":         [ {id, home, away, time_et, hours_to_kickoff}, ... ]
}

A task should exit immediately if BOTH lists are empty.
Usage: python status.py
"""
import json
import os
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-4))

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(ROOT, "data-raw", name), encoding="utf-8") as f:
        return json.load(f)


def knockout_fixtures():
    """Knockout matches (ids 73-104) from bracket.json, with the current
    projected/actual team names pulled from predictions.json when available."""
    try:
        br = load("bracket.json")
    except Exception:
        return []
    names = {}
    try:
        path = os.path.join(ROOT, "data", "predictions.json")
        ko = json.load(open(path, encoding="utf-8")).get("knockout") or {}
        for b in ko.get("bracket", []):
            names[b["id"]] = (b.get("home"), b.get("away"))
    except Exception:
        pass
    out = []
    rounds = br.get("rounds", [])
    extra = [br["third_place"]] if br.get("third_place") else []
    for rd in rounds:
        for m in rd["matches"]:
            h, a = names.get(m["id"], (None, None))
            out.append({"id": m["id"], "date": m.get("date"), "round": rd["name"],
                        "home": h or rd["name"], "away": a or f"match {m['id']}"})
    for m in extra:
        h, a = names.get(m["id"], (None, None))
        out.append({"id": m["id"], "date": m.get("date"), "round": "Third place",
                    "home": h or "Third place", "away": a or f"match {m['id']}"})
    return out


def main():
    sched = load("schedule.json")
    try:
        results = load("results.json") or {}
    except Exception:
        results = {}
    today = datetime.now(ET)
    now = today
    finished, upcoming = [], []
    # group stage (has kickoff times)
    for m in sched["matches"]:
        try:
            ko = datetime.strptime(f"{m['date']} {m['time_et']}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        except (ValueError, KeyError):
            continue
        if str(m["id"]) in results:
            continue
        hrs = (ko - now).total_seconds() / 3600.0
        row = {"id": m["id"], "home": m["home"], "away": m["away"],
               "date": m["date"], "time_et": m.get("time_et")}
        if hrs <= -2.5:                       # kicked off >2.5h ago, no result yet
            finished.append(row)
        elif 0 <= hrs <= 4:                   # kicks off within 4h
            row["hours_to_kickoff"] = round(hrs, 1)
            upcoming.append(row)
    # knockout stage (date only, no kickoff time stored) — coarse day-based gate
    todate = today.date()
    for m in knockout_fixtures():
        if str(m["id"]) in results or not m.get("date"):
            continue
        try:
            d = datetime.strptime(m["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < todate:                        # a past day, still no result
            finished.append(m)
        elif d == todate:                     # plays today
            upcoming.append(m)
    print(json.dumps({"finished_unrecorded": finished, "upcoming_4h": upcoming}))


if __name__ == "__main__":
    main()
