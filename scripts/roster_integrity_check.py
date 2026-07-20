"""Audit the dated roster snapshot without pretending an incomplete feed is final."""
import json
from pathlib import Path

from leagues import config

ROOT = Path(__file__).resolve().parents[1]
ROSTERS = ROOT / "data-raw" / "leagues" / "rosters.json"
CLUBS = {
    "PL": "clubs.json",
    "LALIGA": "clubs_laliga.json",
    "BUNDESLIGA": "clubs_bundesliga.json",
    "LIGUE1": "clubs_ligue1.json",
}


def audit(payload):
    errors, warnings = [], []
    for league, cfg in config.LEAGUES.items():
        teams = payload.get(league, {})
        expected = json.loads(
            (ROOT / "data" / "leagues" / CLUBS[league]).read_text(encoding="utf-8"))
        missing = sorted(set(expected) - set(teams))
        extra = sorted(set(teams) - set(expected))
        if len(teams) != cfg.n_teams or missing or extra:
            errors.append(
                f"{league}: expected {cfg.n_teams} clubs; missing={missing}, extra={extra}")

        seen = {}
        for club, entry in teams.items():
            players = entry.get("players", [])
            if len(players) < 18:
                warnings.append(
                    f"{league}/{club}: only {len(players)} players in source snapshot")
            for player in players:
                pid = player["id"]
                if pid in seen and seen[pid] != club:
                    errors.append(
                        f"{league}: {player['name']} ({pid}) listed for "
                        f"{seen[pid]} and {club}")
                seen[pid] = club
    return errors, warnings


def main():
    payload = json.loads(ROSTERS.read_text(encoding="utf-8"))
    errors, warnings = audit(payload)
    for item in errors:
        print("ERROR:", item)
    for item in warnings:
        print("WARNING:", item)
    print(f"{len(errors)} error(s), {len(warnings)} incomplete-roster warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
