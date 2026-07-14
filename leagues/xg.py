"""Per-match team xG from Understat (via soccerdata, disk-cached)."""
import pandas as pd
import soccerdata as sd

from leagues import config
from leagues.names import canonical

# Understat seasons are the starting year: 2021 => 2021/22
SEASONS = [2021, 2022, 2023, 2024, 2025]


def fetch_team_match_xg(league: str) -> pd.DataFrame:
    """Return one row per team-match: date, team, home flag, xg, xga."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=SEASONS)
    raw = us.read_team_match_stats().reset_index()

    required = ["date", "home_team", "away_team", "home_xg", "away_xg"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(
            f"Understat schema changed; missing {missing}. Got: {list(raw.columns)}"
        )

    raw = raw.assign(date=pd.to_datetime(raw["date"]))

    home_rows = pd.DataFrame({
        "date": raw["date"],
        "team": [canonical(t, league) for t in raw["home_team"]],
        "xg": pd.to_numeric(raw["home_xg"], errors="coerce"),
        "xga": pd.to_numeric(raw["away_xg"], errors="coerce"),
        "is_home": True,
    })
    away_rows = pd.DataFrame({
        "date": raw["date"],
        "team": [canonical(t, league) for t in raw["away_team"]],
        "xg": pd.to_numeric(raw["away_xg"], errors="coerce"),
        "xga": pd.to_numeric(raw["home_xg"], errors="coerce"),
        "is_home": False,
    })

    out = pd.concat([home_rows, away_rows], ignore_index=True)
    out = out.sort_values(["date", "team"]).reset_index(drop=True)
    return out.dropna(subset=["xg"]).reset_index(drop=True)
