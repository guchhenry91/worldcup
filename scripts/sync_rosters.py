"""Snapshot current first-team rosters for every configured league.

ESPN's league/team feeds expose the active 2026-27 competition membership and
current roster in one consistent schema. This snapshot is an audit input, not a
source of performance statistics: Understat remains the rate source.

Run during an open transfer window and before each publish:
    python -m scripts.sync_rosters
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

from leagues.names import canonical

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "leagues" / "rosters.json"
LEAGUES = {
    "PL": "eng.1",
    "LALIGA": "esp.1",
    "BUNDESLIGA": "ger.1",
    "LIGUE1": "fra.1",
}
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "worldcup-roster-audit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


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


def main():
    payload = {
        "_verified_at": datetime.now(timezone.utc).isoformat(),
        "_source": "ESPN 2026-27 league and team roster feeds",
        "_provisional": (
            "The summer registration window is open. This is a dated snapshot, "
            "not a final registered-squad list."
        ),
    }
    for key, slug in LEAGUES.items():
        payload[key] = fetch_league(key, slug)
        print(f"{key}: {len(payload[key])} clubs, "
              f"{sum(len(v['players']) for v in payload[key].values())} players")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
