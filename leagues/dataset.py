"""Assemble the unified per-match training table: results + closing odds + xG."""
import pandas as pd

from leagues import history, xg


def build_matches(league: str) -> pd.DataFrame:
    """One row per match: date, home, away, goals, odds, and (where available)
    home_xg / away_xg. Matches with no xG keep NaN — the model falls back to goals."""
    hist = history.fetch_history(league)
    try:
        tx = xg.fetch_team_match_xg(league)
    except Exception as exc:  # xG is an enhancement, never a hard dependency
        print(f"WARNING: xG unavailable for {league} ({exc}); falling back to goals only")
        hist["home_xg"] = pd.NA
        hist["away_xg"] = pd.NA
        return hist

    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    if getattr(tx["date"].dt, "tz", None) is not None:
        tx["date"] = tx["date"].dt.tz_localize(None)
    tx["day"] = tx["date"].dt.normalize()

    home_x = (tx[tx["is_home"] == True][["day", "team", "xg"]]
              .rename(columns={"team": "home", "xg": "home_xg"}))
    away_x = (tx[tx["is_home"] == False][["day", "team", "xg"]]
              .rename(columns={"team": "away", "xg": "away_xg"}))

    hist = hist.copy()
    hist["day"] = pd.to_datetime(hist["date"]).dt.normalize()
    out = hist.merge(home_x, on=["day", "home"], how="left")
    out = out.merge(away_x, on=["day", "away"], how="left")

    # Exact-day join can miss matches listed on an adjacent day (timezone /
    # late kickoff offsets between football-data.co.uk and Understat). For
    # rows still missing xG, retry with the day shifted +/-1 and fill gaps
    # by index only (never duplicate rows).
    for shift in (1, -1):
        missing_home = out["home_xg"].isna()
        if missing_home.any():
            shifted = home_x.copy()
            shifted["day"] = shifted["day"] + pd.Timedelta(days=shift)
            # drop duplicate (day, home) keys first: a left-merge against a frame
            # with dup keys returns MORE rows than the left side, and the
            # positional index assignment below would then raise a length mismatch.
            shifted = shifted.drop_duplicates(["day", "home"])
            candidate = out.loc[missing_home, ["day", "home"]].merge(
                shifted, on=["day", "home"], how="left"
            )
            candidate.index = out.loc[missing_home].index
            out.loc[missing_home, "home_xg"] = out.loc[missing_home, "home_xg"].fillna(
                candidate["home_xg"]
            )

        missing_away = out["away_xg"].isna()
        if missing_away.any():
            shifted = away_x.copy()
            shifted["day"] = shifted["day"] + pd.Timedelta(days=shift)
            shifted = shifted.drop_duplicates(["day", "away"])
            candidate = out.loc[missing_away, ["day", "away"]].merge(
                shifted, on=["day", "away"], how="left"
            )
            candidate.index = out.loc[missing_away].index
            out.loc[missing_away, "away_xg"] = out.loc[missing_away, "away_xg"].fillna(
                candidate["away_xg"]
            )

    cov = out["home_xg"].notna().mean()
    print(f"{league}: {len(out)} matches, xG coverage {cov:.0%}")
    return out.drop(columns=["day"])
