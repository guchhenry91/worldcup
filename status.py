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


def main():
    sched = load("schedule.json")
    try:
        results = load("results.json") or {}
    except Exception:
        results = {}
    now = datetime.now(ET)
    finished, upcoming = [], []
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
    print(json.dumps({"finished_unrecorded": finished, "upcoming_4h": upcoming}))


if __name__ == "__main__":
    main()
