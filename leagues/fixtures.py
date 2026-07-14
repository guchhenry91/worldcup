"""2026-27 fixtures (and live results) from fixturedownload.com JSON feeds."""
import json
import urllib.request

import pandas as pd

from leagues import config
from leagues.names import canonical

FEED = "https://fixturedownload.com/feed/json/{slug}"


def parse_fixtures(raw: list[dict], league: str) -> pd.DataFrame:
    """Pure parser — takes the decoded JSON list, returns a clean DataFrame."""
    rows = []
    for r in raw:
        hg, ag = r.get("HomeTeamScore"), r.get("AwayTeamScore")
        played = hg is not None and ag is not None
        rows.append({
            "match_id": r["MatchNumber"],
            "round": r["RoundNumber"],
            "date": pd.to_datetime(r["DateUtc"], utc=True),
            "venue": r.get("Location") or "",
            "home": canonical(r["HomeTeam"], league),
            "away": canonical(r["AwayTeam"], league),
            "home_goals": int(hg) if played else pd.NA,
            "away_goals": int(ag) if played else pd.NA,
            "played": played,
        })
    return pd.DataFrame(rows, columns=["match_id", "round", "date", "venue",
                                       "home", "away", "home_goals", "away_goals",
                                       "played"])


def fetch_fixtures(league: str) -> pd.DataFrame:
    """Download the season's fixtures+results for one league."""
    slug = config.get(league).fixture_slug
    req = urllib.request.Request(
        FEED.format(slug=slug),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return parse_fixtures(raw, league)
