"""Snapshot current first-team rosters for every configured league.

ESPN's league/team feeds expose the active 2026-27 competition membership and
current roster in one consistent schema. This snapshot is an audit input, not a
source of performance statistics: Understat remains the rate source.

Run during an open transfer window and before each publish:
    python -m scripts.sync_rosters
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import time
import urllib.request

from leagues.names import canonical
from leagues.api_football import Client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "leagues" / "rosters.json"
LEAGUES = {
    "PL": "eng.1",
    "LALIGA": "esp.1",
    "BUNDESLIGA": "ger.1",
    "LIGUE1": "fra.1",
}
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
REQUEST_ATTEMPTS = 4
REQUEST_TIMEOUT = 20
API_LEAGUES = {"PL": 39, "LALIGA": 140, "BUNDESLIGA": 78, "LIGUE1": 61}
API_SEASON = 2026


def get_json(url, *, attempts=REQUEST_ATTEMPTS, sleeper=time.sleep):
    req = urllib.request.Request(url, headers={"User-Agent": "worldcup-roster-audit/1.0"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == attempts:
                raise
            delay = min(8.0, 2 ** (attempt - 1)) + random.random() * 0.25
            print(f"WARNING: roster request failed ({attempt}/{attempts}); "
                  f"retrying in {delay:.1f}s: {url}")
            sleeper(delay)


def fetch_league(key, slug):
    teams_url = f"{BASE}/{slug}/teams"
    raw = get_json(teams_url)
    teams = raw["sports"][0]["leagues"][0]["teams"]
    result = {}
    for item in teams:
        team = item["team"]
        club = canonical(team["displayName"], key)
        roster_url = f"{BASE}/{slug}/teams/{team['id']}/roster"
        roster = get_json(roster_url)
        players = []
        for athlete in roster.get("athletes", []):
            position = athlete.get("position") or {}
            players.append({
                "id": str(athlete["id"]),
                "name": athlete["displayName"],
                "position": position.get("abbreviation") or position.get("name") or "",
            })
        result[club] = {
            "source": roster_url,
            "players": sorted(players, key=lambda p: p["name"]),
        }
    return result


def _stamp(payload, league):
    return (payload.get("_league_verified_at") or {}).get(
        league, payload.get("_verified_at"))


def _fresh(payload, league, now, hours=36):
    try:
        checked = datetime.fromisoformat(str(_stamp(payload, league)).replace("Z", "+00:00"))
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
        return (now - checked.astimezone(timezone.utc)).total_seconds() < hours * 3600
    except (TypeError, ValueError):
        return False


def fetch_api_league(client, key):
    teams = client.get("teams", league=API_LEAGUES[key], season=API_SEASON)
    result = {}
    for item in teams:
        team = item["team"]
        club = canonical(team["name"], key)
        squad_rows = client.get("players/squads", team=team["id"])
        players = (squad_rows[0].get("players") or []) if squad_rows else []
        result[club] = {
            "source": f"api-football:team:{team['id']}",
            "players": sorted([{
                "id": str(player["id"]),
                "name": player["name"],
                "position": player.get("position") or "",
            } for player in players], key=lambda player: player["name"]),
        }
    return result


def main():
    previous = None
    if OUT.exists():
        try:
            previous = json.loads(OUT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    now = datetime.now(timezone.utc)
    if os.environ.get("API_FOOTBALL_KEY"):
        payload = previous or {}
        payload.setdefault("_league_verified_at", {})
        # Alternate pairs daily. Each league refreshes every 48 hours, costing
        # about 42 calls and preserving more than half the free quota for lineups.
        pairs = (("PL", "BUNDESLIGA"), ("LALIGA", "LIGUE1"))
        selected = pairs[now.toordinal() % 2]
        due = [key for key in selected if not _fresh(payload, key, now)]
        if not due:
            print("API-Football rosters are fresh; no quota used")
            return 0
        client = Client(limit=45)
        try:
            for key in due:
                payload[key] = fetch_api_league(client, key)
                payload["_league_verified_at"][key] = now.isoformat()
                print(f"{key}: {len(payload[key])} clubs refreshed from API-Football")
        except Exception as exc:
            if previous and all(previous.get(key) for key in LEAGUES):
                print(f"WARNING: API-Football roster refresh failed ({exc}); "
                      "retaining the last verified snapshot")
                return 0
            raise
        payload["_verified_at"] = max(payload["_league_verified_at"].values())
        payload["_source"] = "API-Football current squad feeds"
        payload["_provisional"] = (
            "Squads rotate through a quota-aware 48-hour refresh; team news and "
            "confirmed lineups are checked separately near kickoff.")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(OUT)
        print(f"wrote {OUT}; {client.used} API-Football requests used")
        return 0

    payload = {
        "_verified_at": now.isoformat(),
        "_source": "ESPN 2026-27 league and team roster feeds (fallback)",
        "_provisional": (
            "The summer registration window is open. This is a dated snapshot, "
            "not a final registered-squad list."
        ),
    }
    try:
        for key, slug in LEAGUES.items():
            payload[key] = fetch_league(key, slug)
            print(f"{key}: {len(payload[key])} clubs, "
                  f"{sum(len(v['players']) for v in payload[key].values())} players")
    except Exception as exc:
        # A transient failure must not destroy a complete, previously audited
        # snapshot. Keep its original _verified_at so downstream code can still
        # enforce the 72-hour freshness limit honestly.
        if previous and all(previous.get(key) for key in LEAGUES):
            print(f"WARNING: roster refresh unavailable after retries "
                  f"({type(exc).__name__}: {exc}); retaining the last verified snapshot")
            return 0
        raise
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
