"""ClubElo ratings — the cross-league strength prior (free, no key, HTTP only).

CACHED ON DISK. ClubElo serves a ~20k-row CSV over slow plain HTTP, and we may
pull four snapshots plus a per-club history for every unrated club. Uncached that
measured at 1231s — 20.5 minutes, and 99% of a publish. The ratings only change
once a day, so a day-keyed disk cache is both safe and the difference between a
20-minute job and a 2-second one.
"""
import io
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from leagues.names import ALIASES, UnknownTeam, canonical

API = "http://api.clubelo.com/{d}"   # NOTE: http only — clubelo does not serve https
CACHE = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "cache"
TIMEOUT = 8          # ClubElo is either quick or down; a long timeout just stalls the job


class ClubEloUnavailable(Exception):
    """ClubElo resolved nothing at all — the caller must apply its own prior
    rather than silently rating every club at the league median."""


def _cached(key: str, fetch):
    """Read `key` from the day's cache, else fetch and store it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            path.unlink(missing_ok=True)     # corrupt cache entry: refetch
    df = fetch()
    df.to_csv(path, index=False)
    return df


def fetch_elo_snapshot(on: date | None = None) -> pd.DataFrame:
    """Every club's Elo on a date: columns Rank, Club, Country, Level, Elo, From, To."""
    d = (on or date.today()).isoformat()

    def _get():
        with urllib.request.urlopen(API.format(d=d), timeout=TIMEOUT) as resp:
            return pd.read_csv(io.StringIO(resp.read().decode("utf-8")))

    return _cached(f"snapshot-{d}", _get)


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
    slug = club_spelling.replace(" ", "")

    def _get():
        url = CLUB_API.format(club=urllib.parse.quote(slug))
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            return pd.read_csv(io.StringIO(resp.read().decode("utf-8")))

    try:
        df = _cached(f"club-{slug}-{date.today().isoformat()}", _get)
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
    snapshots_ok = 0
    for back in FALLBACK_DAYS:
        if len(found) == len(want):
            break
        try:
            _harvest(fetch_elo_snapshot(base - timedelta(days=back)), league, want, found)
            snapshots_ok += 1
        except Exception as exc:                     # a bad snapshot must not be fatal
            print(f"WARNING: ClubElo snapshot {back}d back failed: {exc}")

    if not snapshots_ok:
        # Every snapshot failed: the service is down, not merely missing a club.
        # Attempting a per-club history for all 20 clubs would just burn another
        # 20 minutes of timeouts, so give up now.
        raise ClubEloUnavailable(
            f"no ClubElo snapshot resolved for {league} ({len(FALLBACK_DAYS)} tried)")

    missing = sorted(want - set(found))
    if missing:
        found |= _rescue_missing(league, missing)
        missing = sorted(want - set(found))

    if not found:
        # Total outage (observed 2026-07-14: api.clubelo.com timed out on both
        # http and https). Returning a median for EVERY club would hand promoted
        # sides league-average strength and silently mis-rate the whole table --
        # so refuse, and let the caller apply an explicit fallback prior.
        raise ClubEloUnavailable(
            f"ClubElo returned nothing for {league}: no snapshot and no per-club "
            f"history resolved. Refusing to rate every club at the median.")

    if missing:
        print(f"WARNING: {league}: no ClubElo rating for {missing} — using league median")
    median = float(pd.Series(list(found.values())).median())
    return {t: found.get(t, median) for t in teams}
