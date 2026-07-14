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
import pandas as pd
import soccerdata as sd

from leagues import config
from leagues.names import canonical

# Understat "result" values that count as on target. A shot against the post is
# NOT on target, and an own goal is not the shooter's shot at all.
ON_TARGET = {"Goal", "Saved Shot"}
POSITION_MAP = {"F": "FW", "M": "MF", "D": "DF", "GK": "GK", "AM": "AM"}


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


def build_player_logs(season_stats: pd.DataFrame, shots: pd.DataFrame,
                      league: str) -> pd.DataFrame:
    """Pure parser -- no network. One row per player-season."""
    df = season_stats.copy()
    df["team"] = [canonical(t, league) for t in df["team"]]
    df["pos"] = [understat_position(p) for p in df["position"]]
    for c in ("minutes", "np_goals", "np_xg", "shots"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.rename(columns={"np_xg": "npxg"})

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
    # half-players at two different clubs.
    latest = (df.sort_values("season").groupby("player")["team"].last())
    df["team"] = df["player"].map(latest)

    df["date"] = [season_end(s) for s in df["season"]]
    return df[["date", "season", "team", "player", "pos", "minutes", "np_goals",
               "shots", "sot", "npxg", "pens_att"]].reset_index(drop=True)


def fetch_player_logs(league: str) -> pd.DataFrame:
    """Download (cached) five seasons of player season totals + shot events."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=list(lg.history_seasons))

    stats = us.read_player_season_stats().reset_index()
    required = {"season", "team", "player", "position", "minutes", "np_goals",
                "np_xg", "shots"}
    missing = required - set(stats.columns)
    if missing:
        raise RuntimeError(f"Understat player schema changed; missing {sorted(missing)}. "
                           f"Got: {list(stats.columns)}")

    shots = us.read_shot_events().reset_index()
    for col in ("season", "team", "player", "result", "situation"):
        if col not in shots.columns:
            raise RuntimeError(f"Understat shot schema changed; missing {col!r}. "
                               f"Got: {list(shots.columns)}")

    return build_player_logs(stats, shots, league)


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
