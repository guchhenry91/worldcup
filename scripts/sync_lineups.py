"""Fetch confirmed XIs shortly before kickoff without wasting the daily quota."""
import json
import os
from datetime import datetime, timezone

from leagues.api_football import Client
from leagues.names import canonical, UnknownTeam
from leagues.team_news import NEWS_PATH, upcoming_fixtures
from scripts.sync_rosters import API_LEAGUES


def main(now=None):
    if not os.environ.get("API_FOOTBALL_KEY"):
        print("API_FOOTBALL_KEY is not set; skipping confirmed lineups")
        return 0
    now = now or datetime.now(timezone.utc)
    imminent = []
    for fixture in upcoming_fixtures(now=now):
        kickoff = datetime.fromisoformat(fixture["date"].replace("Z", "+00:00"))
        minutes = (kickoff - now).total_seconds() / 60
        if 0 < minutes <= 40:
            imminent.append((fixture, kickoff))
    if not imminent:
        print("no fixtures inside the 55-minute lineup window; no quota used")
        return 0

    news = json.loads(NEWS_PATH.read_text(encoding="utf-8")) if NEWS_PATH.exists() else {}
    client = Client(limit=44)
    by_date = {}
    changed = False
    confirmed = 0
    for fixture, kickoff in imminent:
        league = fixture["league_key"]
        section = news.setdefault(league, {})
        if all((section.get(team) or {}).get("lineup_confirmed") is True
               for team in (fixture["home"], fixture["away"])):
            continue
        date = kickoff.date().isoformat()
        if date not in by_date:
            by_date[date] = client.get("fixtures", date=date, timezone="UTC")
        match = None
        for candidate in by_date[date]:
            if candidate.get("league", {}).get("id") != API_LEAGUES[league]:
                continue
            try:
                home = canonical(candidate["teams"]["home"]["name"], league)
                away = canonical(candidate["teams"]["away"]["name"], league)
            except UnknownTeam:
                continue
            if (home, away) == (fixture["home"], fixture["away"]):
                match = candidate
                break
        if not match:
            print(f"WARNING: API-Football fixture not matched: {league} "
                  f"{fixture['home']} v {fixture['away']}")
            continue
        fixture_id = match["fixture"]["id"]
        if any((section.get(team) or {}).get("lineup_api_attempted_fixture") == fixture_id
               for team in (fixture["home"], fixture["away"])):
            continue
        lineups = client.get("fixtures/lineups", fixture=fixture_id)
        # One attempt per fixture. This marker is committed even when the provider
        # has no XI yet, preventing the next 30-minute run from spending the same
        # daily quota again. The 40-minute window matches the provider's documented
        # publication timing.
        for team in (fixture["home"], fixture["away"]):
            section.setdefault(team, {}).update({
                "lineup_api_attempted_fixture": fixture_id,
                "lineup_api_attempted_at": now.isoformat(),
            })
        changed = True
        if len(lineups) != 2:
            print(f"lineup not published yet: {fixture['home']} v {fixture['away']}")
            continue
        for row in lineups:
            try:
                team = canonical(row["team"]["name"], league)
            except UnknownTeam:
                continue
            starters = [item["player"]["name"] for item in row.get("startXI", [])]
            bench = [item["player"]["name"] for item in row.get("substitutes", [])]
            if len(starters) != 11:
                continue
            entry = section.setdefault(team, {})
            entry.update({"starters": starters, "bench": bench,
                          "lineup_confirmed": True,
                          "lineup_checked_at": now.isoformat(),
                          "lineup_source": "API-Football",
                          "lineup_fixture_id": fixture_id})
            confirmed += 1
    if changed:
        tmp = NEWS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(news, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(NEWS_PATH)
    print(f"confirmed {confirmed} team lineups; {client.used} API-Football requests used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
