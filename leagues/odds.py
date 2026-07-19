"""Live 1X2 market odds for UPCOMING fixtures, from football-data.co.uk's
fixtures.csv -- the SAME feed the historical backtest is scored against, so the
de-vig (leagues.backtest.devig) and team-name mapping (leagues.names) are reused
unchanged.

OFF-SEASON: fixtures.csv only lists ~1 week ahead, so before ~mid-August it has
no top-flight rows and parse_fixture_odds returns an empty (well-formed) frame.
This must NEVER raise or block a publish -- a fixture with no odds simply gets no
market line on its card.
"""
import io
import urllib.request

import pandas as pd

from leagues.backtest import devig
from leagues.names import canonical, UnknownTeam

FEED = "https://www.football-data.co.uk/fixtures.csv"
DIV = {"PL": "E0", "LALIGA": "SP1", "BUNDESLIGA": "D1", "LIGUE1": "F1"}
COLS = ["date", "home", "away", "m_home", "m_draw", "m_away"]


def parse_fixture_odds(buf, league: str) -> pd.DataFrame:
    """Pure parser (no network). `buf` is a file-like CSV; returns de-vigged 1X2
    per upcoming fixture for one league."""
    raw = pd.read_csv(buf)
    if "Div" not in raw.columns:
        return pd.DataFrame(columns=COLS)
    raw = raw[raw["Div"] == DIV[league]]
    rows = []
    for _, r in raw.iterrows():
        h, d, a = r.get("AvgH"), r.get("AvgD"), r.get("AvgA")
        # Skip anything that is not a usable decimal price. NaN was already handled;
        # ZERO was not, and 1.0/0 raised inside this loop, escaped to the blanket
        # handler in fetch_fixture_odds, and silently dropped the WHOLE league's
        # market lines while blaming the network. One bad cell must cost one
        # fixture its line, never the league its feed.
        try:
            h, d, a = float(h), float(d), float(a)
        except (TypeError, ValueError):
            continue
        if not (h > 1.0 and d > 1.0 and a > 1.0):
            continue                          # not priced yet -> no line, not an error
        try:
            home = canonical(r["HomeTeam"], league)
            away = canonical(r["AwayTeam"], league)
        except UnknownTeam:
            continue                          # unmapped spelling -> skip, never crash a publish
        ph, pdw, pa = devig(h, d, a)
        rows.append({"date": r.get("Date"), "home": home, "away": away,
                     "m_home": ph, "m_draw": pdw, "m_away": pa})
    return pd.DataFrame(rows, columns=COLS)


def fetch_fixture_odds(league: str) -> pd.DataFrame:
    """Download the upcoming-fixtures odds feed for one league. Returns an empty
    frame off-season or on any network failure -- never raises."""
    req = urllib.request.Request(FEED, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            text = resp.read().decode("utf-8", "replace")
        return parse_fixture_odds(io.StringIO(text), league)
    except Exception as exc:
        print(f"odds feed unavailable for {league} ({exc}); no market lines this run")
        return pd.DataFrame(columns=COLS)


def market_for(odds: pd.DataFrame, home: str, away: str) -> dict | None:
    """Look up one fixture's de-vigged market line by canonical home+away."""
    if odds.empty:
        return None
    hit = odds[(odds["home"] == home) & (odds["away"] == away)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {"p_home": round(float(r["m_home"]), 3),
            "p_draw": round(float(r["m_draw"]), 3),
            "p_away": round(float(r["m_away"]), 3)}
