"""Player data from Understat: season totals + shot events.

WHY NOT FBREF: soccerdata's FBref player-match reader drives a headless Chrome
once per match page (~4/min), so five seasons of one league is ~8 hours and four
leagues is days. Understat gives the same signal in seconds:

  read_player_season_stats  -- minutes, non-penalty goals, npxG, shots, position
                               (one request per season)
  read_shot_events          -- every shot: player, result, situation
                               (one request per season; ~25s for 9.5k shots)

Shots on target and penalty attempts are not in the season totals, so they are
derived from the shot events. Penalty attempts also tell us empirically WHO takes
the penalties, rather than us hardcoding a taker per club.

Output is one row per player-SEASON (not per appearance). props.player_rates
sums over rows and weights them by age, so season rows and appearance rows carry
identical semantics -- the only thing lost is per-match granularity, which the
props gate works around (see props_backtest.py).
"""
import json
from pathlib import Path
import unicodedata

import pandas as pd
import soccerdata as sd

from leagues import config
from leagues.names import canonical, UnknownTeam

# Understat "result" values that count as on target. A shot against the post is
# NOT on target, and an own goal is not the shooter's shot at all.
ON_TARGET = {"Goal", "Saved Shot"}
POSITION_MAP = {"F": "FW", "M": "MF", "D": "DF", "GK": "GK", "AM": "AM"}
MIN_COMPLETE_ROSTER = 18
MAX_ROSTER_AGE_HOURS = 72


def _player_key(name: str) -> str:
    """Accent/case/punctuation-insensitive player identity key.

    This is deliberately stricter than fuzzy matching: a false match can assign a
    departed player to the wrong club and produce a confident-looking prop.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return "".join(c for c in text if c.isalnum())


def load_roster_snapshot(league: str) -> dict:
    """Load the dated free-source roster snapshot for one league."""
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "rosters.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get(league, {})


def roster_snapshot_age_hours() -> float | None:
    """Age of the roster evidence, or None when absent/malformed."""
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "rosters.json"
    if not path.exists():
        return None
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))["_verified_at"]
        checked = pd.Timestamp(stamp)
        checked = (checked.tz_localize("UTC") if checked.tzinfo is None
                   else checked.tz_convert("UTC"))
        return float((pd.Timestamp.now("UTC") - checked).total_seconds() / 3600)
    except (KeyError, TypeError, ValueError):
        return None


def reconcile_rates_to_roster(rates: pd.DataFrame, league: str,
                              min_players: int = MIN_COMPLETE_ROSTER):
    """Return (safe rates, incomplete clubs, unmatched historical players).

    Clubs with a thin source roster are removed entirely: publishing no player
    market is safer than treating a partial list as the squad. For complete clubs,
    only exact normalized roster members survive and their current club assignment
    comes from the snapshot rather than their last historical Understat season.
    """
    snapshot = load_roster_snapshot(league)
    age = roster_snapshot_age_hours()
    if not snapshot or age is None or age > MAX_ROSTER_AGE_HOURS:
        teams = sorted(snapshot) if snapshot else (
            sorted(set(rates["team"])) if not rates.empty else [])
        return rates.iloc[0:0].copy(), teams, []

    incomplete = sorted(
        club for club, entry in snapshot.items()
        if len(entry.get("players", [])) < min_players
    )
    current = {}
    duplicate_keys = set()
    for club, entry in snapshot.items():
        if club in incomplete:
            continue
        for player in entry.get("players", []):
            key = _player_key(player.get("name", ""))
            if not key:
                continue
            if key in current and current[key] != club:
                duplicate_keys.add(key)
            current[key] = club
    for key in duplicate_keys:
        current.pop(key, None)  # ambiguous identity -> withhold, never guess

    kept, unmatched = [], []
    for _, row in rates.iterrows():
        club = current.get(_player_key(row["player"]))
        if club is None:
            unmatched.append(f"{row['team']}/{row['player']}")
            continue
        item = row.copy()
        item["team"] = club
        kept.append(item)
    safe = pd.DataFrame(kept, columns=rates.columns)
    return safe.reset_index(drop=True), incomplete, sorted(unmatched)


def understat_position(pos: str) -> str:
    """Understat spells positions like "F M S" / "D S" / "GK"; "S" means
    substitute, so take the first token that is a real position."""
    for token in str(pos).split():
        if token in POSITION_MAP:
            return POSITION_MAP[token]
    return "MF"


def season_end(season: str) -> pd.Timestamp:
    """"2526" -> 2026-05-31. The decay in props.player_rates is measured from
    when the football was played, so each season row is dated at its end."""
    end_year = 2000 + int(str(season)[2:4])
    return pd.Timestamp(year=end_year, month=5, day=31)


def _assign_current_club(df: pd.DataFrame, transfers: dict | None) -> pd.DataFrame:
    """Attribute every player to his CURRENT club and drop players who left.

    A player's club is his most recent SEASON's club -- but Understat has no data
    for the in-progress season, so summer-window moves are invisible. `transfers`
    (player -> new canonical club, or None if he left the league) is a manual
    override applied on top: it re-attributes a moved player's whole history to
    his new club (his scoring rate follows him) and removes anyone who left."""
    latest = df.sort_values("season").groupby("player")["team"].last()
    for player, club in (transfers or {}).items():
        latest[player] = club                        # club may be None (departed)
    df = df.copy()
    df["team"] = df["player"].map(latest)
    return df[df["team"].notna()].reset_index(drop=True)   # drop departed players


def build_player_logs(season_stats: pd.DataFrame, shots: pd.DataFrame,
                      league: str, transfers: dict | None = None) -> pd.DataFrame:
    """Pure parser -- no network. One row per player-season."""
    df = season_stats.copy()
    df["team"] = [canonical(t, league) for t in df["team"]]
    df["pos"] = [understat_position(p) for p in df["position"]]
    for c in ("minutes", "np_goals", "np_xg", "shots"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Understat exposes appearances as `matches`. Keep them: event probability
    # needs to distinguish "50% chance of 80 minutes" from "certain to play 40".
    # They have the same expected minutes but very different chances of 2+ shots.
    if "matches" in df:
        df["appearances"] = pd.to_numeric(df["matches"], errors="coerce").fillna(0)
    else:
        # Compatibility with older cached frames. This conservative estimate
        # never claims more appearances than full-match equivalents observed.
        df["appearances"] = (df["minutes"] / 90.0).clip(lower=0)
    df = df.rename(columns={"np_xg": "npxg"})

    if shots is None or len(shots) == 0:
        # Shot-level data unavailable (an upstream soccerdata parser bug crashes
        # read_shot_events on some leagues -- one match returns its roster as a
        # list, not a dict). Degrade rather than lose the whole league: estimate
        # SOT from the league-average on-target ratio, and leave penalties unknown.
        from leagues.props import SOT_RATIO_PRIOR
        df["season"] = df["season"].astype(str)
        df["sot"] = (df["shots"] * SOT_RATIO_PRIOR).round().astype(int)
        df["pens_att"] = 0
        df = _assign_current_club(df, transfers)
        df["date"] = [season_end(s) for s in df["season"]]
        return df[["date", "season", "team", "player", "pos", "minutes", "appearances", "np_goals",
                   "shots", "sot", "npxg", "pens_att"]].reset_index(drop=True)

    # shots on target + penalty attempts, per player-season, from the shot events
    ev = shots.copy()
    ev["team"] = [canonical(t, league) for t in ev["team"]]
    ev["is_sot"] = ev["result"].isin(ON_TARGET)
    # PENALTIES ARE NOT LABELLED. soccerdata's Understat reader maps the source's
    # "Penalty" situation to NA rather than a value, so `situation` is null for
    # exactly the penalties and nothing else. Verified on 2025-26 PL: all 92
    # NA-situation shots fall in the 0.70-0.80 xG band (penalty xG is ~0.76) and
    # the players taking them are the actual PL penalty takers. Matching on NA is
    # therefore correct -- and matching on the string "penalty" silently yields
    # zero takers, which is the bug this comment exists to prevent.
    ev["is_pen"] = ev["situation"].isna()
    agg = (ev.groupby(["season", "player"], as_index=False)
             .agg(sot=("is_sot", "sum"), pens_att=("is_pen", "sum")))
    agg["season"] = agg["season"].astype(str)
    df["season"] = df["season"].astype(str)
    df = df.merge(agg, on=["season", "player"], how="left")
    df["sot"] = df["sot"].fillna(0).astype(int)
    df["pens_att"] = df["pens_att"].fillna(0).astype(int)

    # A player who changed clubs must be attributed to his CURRENT club, or
    # props.player_rates (which groups by team+player) would split him into two
    # half-players at two different clubs. Transfer overrides are applied here too.
    df = _assign_current_club(df, transfers)

    df["date"] = [season_end(s) for s in df["season"]]
    return df[["date", "season", "team", "player", "pos", "minutes", "appearances", "np_goals",
               "shots", "sot", "npxg", "pens_att"]].reset_index(drop=True)


def fetch_player_logs(league: str, apply_transfers: bool = True) -> pd.DataFrame:
    """Download (cached) five seasons of player season totals + shot events.

    apply_transfers=False returns the RAW attribution with no overrides -- used by
    scripts/apply_transfers.py so it can still find players a previous override
    removed (otherwise resolving names against an already-filtered list would drop
    them from the file and silently restore them to their old club)."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=list(lg.history_seasons))

    stats = us.read_player_season_stats().reset_index()
    required = {"season", "team", "player", "position", "minutes", "np_goals",
                "np_xg", "shots"}
    missing = required - set(stats.columns)
    if missing:
        raise RuntimeError(f"Understat player schema changed; missing {sorted(missing)}. "
                           f"Got: {list(stats.columns)}")

    try:
        shots = us.read_shot_events().reset_index()
        for col in ("season", "team", "player", "result", "situation"):
            if col not in shots.columns:
                raise RuntimeError(f"Understat shot schema changed; missing {col!r}. "
                                   f"Got: {list(shots.columns)}")
    except Exception as exc:
        # Upstream soccerdata bug (e.g. GER-Bundesliga: a match roster comes back
        # as a list, not a dict, crashing read_shot_events). Don't sink the whole
        # league -- degrade to season stats only (see build_player_logs).
        print(f"WARNING: shot events unavailable for {league} ({type(exc).__name__}: "
              f"{exc}); shots-on-target use the league-average ratio and penalty "
              f"takers are not identified")
        shots = None

    tr = load_transfers(league) if apply_transfers else None
    return build_player_logs(stats, shots, league, transfers=tr)


def match_player_stats(league: str, seasons=None) -> pd.DataFrame:
    """Per-player, per-MATCH actuals -- the feed player picks are graded against.

    fetch_player_logs is one row per player-SEASON: right for rates, useless for
    grading, because a season total cannot say whether a man scored in a given
    fixture. Shot events are the only per-match player data available here, so
    goals/shots/SOT are counted from them directly.

    Goals count PENALTIES: an anytime-scorer pick wins on a penalty, and grading
    on np_goals would mark a penalty-only scorer wrong when the pick actually won.
    Own goals are excluded from every column -- they are credited to the scorer but
    are not a shot for his own team, and they never settle a scorer bet.

    Returns date, game_id, team, player, goals, shots, sot. Returns an EMPTY frame
    if shot events cannot be read (the known upstream Bundesliga crash), so callers
    leave those picks PENDING rather than grading them all wrong.
    """
    lg = config.get(league)
    seasons = list(seasons) if seasons else list(lg.history_seasons)
    try:
        us = sd.Understat(leagues=lg.understat, seasons=seasons)
        ev = us.read_shot_events().reset_index()
    except Exception as exc:
        print(f"WARNING: no per-match player data for {league} "
              f"({type(exc).__name__}: {exc}); player picks stay PENDING")
        return pd.DataFrame(columns=["date", "game_id", "team", "player",
                                     "goals", "shots", "sot"])

    ev = ev[ev["result"] != "Own Goal"].copy()
    ev["team"] = [canonical(t, league) for t in ev["team"]]
    ev["date"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    ev["is_goal"] = ev["result"] == "Goal"
    ev["is_sot"] = ev["result"].isin(ON_TARGET)
    out = (ev.groupby(["game_id", "date", "team", "player"], as_index=False)
             .agg(goals=("is_goal", "sum"), shots=("result", "size"),
                  sot=("is_sot", "sum")))
    for c in ("goals", "shots", "sot"):
        out[c] = out[c].astype(int)
    return out


def load_news(league: str) -> dict:
    """Team news for one league: club -> {"out": [...], "doubt": [...], ...}.

    Injuries, suspensions and confirmed-XI omissions, gathered per matchweek for
    Best Picks fixtures only (see docs/superpowers/specs/2026-07-19-leagues-team-
    news-design.md). Absent file or league -> {}, and the props are then built from
    squad history alone exactly as before.

    Club names are canonicalised here so the file can use ordinary spellings.
    """
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "news.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8")).get(league, {})
    out = {}
    for club, entry in raw.items():
        try:
            out[canonical(club, league)] = entry
        except UnknownTeam:
            print(f"WARNING: news.json has unmapped club {club!r} for {league}")
    return out


def news_unavailable(news: dict, teams) -> tuple[set, set]:
    """(players ruled out, players doubtful) across the given clubs."""
    out, doubt = set(), set()
    for t in teams:
        e = news.get(t) or {}
        out.update(e.get("out") or [])
        doubt.update(e.get("doubt") or [])
    return out, doubt


def lineup_players(news: dict, teams) -> tuple[set, set]:
    """Confirmed starters and bench players across the requested clubs.

    A lineup is only trusted when `lineup_confirmed` is true. Predicted XIs may
    still live in the file for display/research, but they must never turn a
    provisional player pick into a locked one.
    """
    starters, bench = set(), set()
    for team in teams:
        entry = news.get(team) or {}
        if entry.get("lineup_confirmed") is not True:
            continue
        starters.update(entry.get("starters") or [])
        bench.update(entry.get("bench") or [])
    return starters, bench


def lineups_confirmed(news: dict, teams) -> bool:
    """True only when every club has an explicitly confirmed XI."""
    teams = tuple(teams)
    return bool(teams) and all(
        (news.get(team) or {}).get("lineup_confirmed") is True
        and len((news.get(team) or {}).get("starters") or []) == 11
        for team in teams
    )


def news_checked_age_hours(news: dict, teams) -> float | None:
    """Hours since the OLDEST of these clubs was news-checked; None if any is
    unchecked. Used to fail loudly rather than publish a stale Best Pick."""
    import datetime as dt
    stamps = []
    for t in teams:
        c = (news.get(t) or {}).get("checked")
        if not c:
            return None
        try:
            stamps.append(pd.Timestamp(c).tz_convert("UTC") if pd.Timestamp(c).tzinfo
                          else pd.Timestamp(c).tz_localize("UTC"))
        except Exception:
            return None
    if not stamps:
        return None
    now = pd.Timestamp.now("UTC")
    return float((now - min(stamps)).total_seconds() / 3600.0)


def transfers_age_days() -> int | None:
    """Days since squads were last verified against transfer news, or None if the
    window is shut (outside ~10 Jun - 2 Sep) or no date is recorded."""
    import datetime as dt
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "transfers.json"
    if not path.exists():
        return None
    today = dt.date.today()
    if not ((6, 10) <= (today.month, today.day) <= (9, 2)):
        return None                                  # window shut: rosters are stable
    checked = json.loads(path.read_text(encoding="utf-8")).get("_verified_on")
    try:
        return (today - dt.date.fromisoformat(checked)).days
    except (TypeError, ValueError):
        return None


def load_transfers(league: str) -> dict:
    """Manual current-window transfer overrides for one league: player -> new
    canonical club (or None if he left the league).

    Understat only has completed seasons, so summer-window moves are invisible
    until real 2026-27 games exist. This file (data-raw/leagues/transfers.json)
    carries verified moves so the props show players at their CURRENT club. Keyed
    by league; player names must match the Understat spelling; club names are
    canonicalised here. Absent file or league -> no overrides."""
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "transfers.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8")).get(league, {})
    out = {}
    for player, club in raw.items():
        out[player] = canonical(club, league) if club else None
    return out


def shot_events_available(league: str) -> bool:
    """Whether shot-level data could be read (False -> SOT/pens are degraded).
    Used by publish to surface an honest data_warning."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=list(lg.history_seasons))
    try:
        us.read_shot_events()
        return True
    except Exception:
        return False


def penalty_takers(logs: pd.DataFrame) -> dict:
    """team -> the player with the most recent-weighted penalty attempts.

    Empirical, not hardcoded: whoever has actually been taking them.
    """
    if logs.empty or logs["pens_att"].sum() == 0:
        return {}
    recent = logs.sort_values("season").copy()
    # weight later seasons far more heavily -- penalty duty changes hands
    rank = recent["season"].rank(method="dense")
    recent["w_pens"] = recent["pens_att"] * (2.0 ** rank)
    tally = (recent.groupby(["team", "player"], as_index=False)["w_pens"].sum()
                   .sort_values("w_pens", ascending=False))
    out = {}
    for _, r in tally.iterrows():
        if r["w_pens"] > 0 and r["team"] not in out:
            out[r["team"]] = r["player"]
    return out


def team_shot_context(league: str, recent_seasons: int = 2) -> dict:
    """How many shots each club takes and CONCEDES per match, vs league average.

    Feeds props.match_props's opp_shot_factor: a player faces more shooting
    opportunity against a club that concedes a lot of shots. Uses only the most
    recent seasons -- shot volume is a tactical property and goes stale fast.

    Returns {"concede_factor": {team: x}, "pens_per_team_match": float}.
    """
    lg = config.get(league)
    seasons = list(lg.history_seasons)[-recent_seasons:]
    us = sd.Understat(leagues=lg.understat, seasons=seasons)
    try:
        ev = us.read_shot_events().reset_index()
    except Exception as exc:
        # Same upstream soccerdata bug that fetch_player_logs guards against (a
        # match roster comes back as a list, not a dict). Degrade to neutral:
        # every opponent concedes at the league average (factor 1.0) and a
        # typical penalty rate. No shot-volume tilt, but the league still builds.
        print(f"WARNING: shot context unavailable for {league} "
              f"({type(exc).__name__}); using neutral opponent factors")
        # pens_per_team_match MUST be 0 here: without shot events fetch_player_logs
        # also degrades and identifies no penalty taker, so any penalty budget we
        # subtract from open play would be assigned to nobody, leaving each team's
        # player goal lambdas summing below the team lambda (deflated scorers).
        return {"concede_factor": {}, "pens_per_team_match": 0.0}
    ev["team"] = [canonical(t, league) for t in ev["team"]]

    # shots conceded = shots taken by the OTHER team in the same game
    per_game = ev.groupby(["game", "team"], as_index=False).size()
    conceded = []
    for game, g in per_game.groupby("game"):
        if len(g) != 2:
            continue                       # a game where one side had no shots at all
        for i, row in g.iterrows():
            other = g[g["team"] != row["team"]]["size"].sum()
            conceded.append({"team": row["team"], "conceded": other})
    c = pd.DataFrame(conceded)
    if c.empty:
        return {"concede_factor": {}, "pens_per_team_match": 0.12}

    rate = c.groupby("team")["conceded"].mean()
    league_avg = float(rate.mean()) or 1.0
    factor = (rate / league_avg).to_dict()

    n_team_matches = len(c)
    pens = int(ev["situation"].isna().sum())
    return {"concede_factor": {k: float(v) for k, v in factor.items()},
            "pens_per_team_match": pens / n_team_matches if n_team_matches else 0.12}


def expected_minutes(logs: pd.DataFrame, matches_per_season: int = 38) -> dict:
    """player -> expected minutes in the next match, from his LATEST season.

    Without this every player who has appeared for the club in five seasons is
    assumed to play 90 minutes, so a squad of ~50 (including players long gone)
    shares out the team's expected goals and the real strikers are crushed down
    to a few percent. Minutes-per-team-match is what actually distributes goals.
    """
    if logs.empty:
        return {}
    latest = logs.sort_values("season").groupby("player").last()
    mins = (latest["minutes"] / matches_per_season).clip(upper=90.0)
    return {p: float(m) for p, m in mins.items()}


def playing_time(logs: pd.DataFrame, matches_per_season: int = 38) -> dict:
    """Player availability and workload as separate quantities.

    Returns player -> {appearance_prob, minutes_if_playing, expected_minutes}.
    Using the latest season keeps tactical role current. A small beta prior stops
    one appearance from becoming 100% availability and keeps probabilities away
    from brittle zero/one extremes until a confirmed lineup overrides them.
    """
    if logs.empty:
        return {}
    latest = logs.sort_values("season").groupby("player").last()
    if "appearances" in latest:
        apps = pd.to_numeric(latest["appearances"], errors="coerce").fillna(0)
    else:
        apps = (latest["minutes"] / 90.0).clip(lower=0)
    apps = apps.clip(lower=0, upper=matches_per_season)
    # Beta(1, 1) smoothing: transparent and deliberately mild.
    p_app = ((apps + 1.0) / (matches_per_season + 2.0)).clip(0.05, 0.98)
    conditional = (latest["minutes"] / apps.replace(0, pd.NA)).fillna(0).clip(0, 90)
    out = {}
    for player in latest.index:
        p = float(p_app.loc[player])
        mins = float(conditional.loc[player])
        out[player] = {
            "appearance_prob": p,
            "minutes_if_playing": mins,
            "expected_minutes": p * mins,
        }
    return out


def current_squad(logs: pd.DataFrame) -> set:
    """Players who appeared in the most recent season -- i.e. are plausibly still
    at the club. Everyone else is a five-seasons-ago ghost who would otherwise
    soak up a share of the team's expected goals."""
    if logs.empty:
        return set()
    newest = logs["season"].max()
    return set(logs[(logs["season"] == newest) & (logs["minutes"] > 0)]["player"])
