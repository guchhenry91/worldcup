"""ClubElo ratings — the cross-league strength prior (free, no key, HTTP only)."""
import io
import urllib.parse
import urllib.request
from datetime import date, timedelta

import pandas as pd

from leagues.names import ALIASES, UnknownTeam, canonical

API = "http://api.clubelo.com/{d}"   # NOTE: http only — clubelo does not serve https


def fetch_elo_snapshot(on: date | None = None) -> pd.DataFrame:
    """Every club's Elo on a date: columns Rank, Club, Country, Level, Elo, From, To."""
    d = (on or date.today()).isoformat()
    with urllib.request.urlopen(API.format(d=d), timeout=30) as resp:
        buf = io.StringIO(resp.read().decode("utf-8"))
    return pd.read_csv(buf)


CLUB_API = "http://api.clubelo.com/{club}"
FALLBACK_DAYS = (0, 7, 30, 90)   # a daily snapshot can be missing clubs; look back


def _harvest(snap: pd.DataFrame, league: str, teams: set[str],
             into: dict[str, float]) -> None:
    """Pull any wanted teams out of one snapshot into `into` (first hit wins)."""
    for _, row in snap.iterrows():
        try:
            name = canonical(str(row["Club"]), league)
        except UnknownTeam:
            continue
        if name in teams and name not in into:
            into[name] = float(row["Elo"])


def fetch_club_latest_elo(club_spelling: str) -> float | None:
    """Latest Elo for one club from its own history endpoint.

    ClubElo drops a club from the DAILY snapshot once its rating window lapses
    (observed: Bayern absent from 2026-07-14, window ended 2026-07-03) — but the
    per-club history still has it. Without this, Bayern would fall back to the
    league median (~1670) instead of its true 2001: a 330-point error on the
    strongest team in the league.
    """
    try:
        url = CLUB_API.format(club=urllib.parse.quote(club_spelling.replace(" ", "")))
        with urllib.request.urlopen(url, timeout=30) as resp:
            df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
    except Exception:
        return None
    df = df.dropna(subset=["Elo"])
    if df.empty:
        return None
    df = df.assign(From=pd.to_datetime(df["From"], errors="coerce")).dropna(subset=["From"])
    if df.empty:
        return None
    return float(df.sort_values("From").iloc[-1]["Elo"])


def _rescue_missing(league: str, missing: list[str]) -> dict[str, float]:
    """For teams absent from every snapshot, try their per-club history, using
    each known spelling (canonical first, then aliases) until one resolves."""
    rescued: dict[str, float] = {}
    table = ALIASES.get(league, {})
    for team in missing:
        for spelling in [team, *sorted(table.get(team, ()))]:
            elo = fetch_club_latest_elo(spelling)
            if elo is not None:
                rescued[team] = elo
                print(f"  rescued {team} from ClubElo history as {spelling!r}: {elo:.0f}")
                break
    return rescued


def elo_for_league(league: str, teams: list[str], on: date | None = None) -> dict[str, float]:
    """Map our canonical team names -> ClubElo rating.

    ClubElo's daily snapshot sometimes omits clubs (observed: Bayern and Stuttgart
    absent on 2026-07-14 but present days either side). Falling back to the league
    median for a missing club would rate Bayern as average and silently corrupt the
    model, so we walk back through earlier snapshots first and only use the median
    as a genuine last resort.
    """
    want = set(teams)
    base = on or date.today()
    found: dict[str, float] = {}
    for back in FALLBACK_DAYS:
        if len(found) == len(want):
            break
        try:
            _harvest(fetch_elo_snapshot(base - timedelta(days=back)), league, want, found)
        except Exception as exc:                     # a bad snapshot must not be fatal
            print(f"WARNING: ClubElo snapshot {back}d back failed: {exc}")
    missing = sorted(want - set(found))
    if missing:
        found |= _rescue_missing(league, missing)
        missing = sorted(want - set(found))
    if missing:
        print(f"WARNING: {league}: no ClubElo rating for {missing} — using league median")
    median = float(pd.Series(list(found.values())).median()) if found else 1500.0
    return {t: found.get(t, median) for t in teams}
