"""Calibrate the second-tier -> top-flight level gap, per league.

For every club promoted in the last few seasons, measure how much its second-tier
goal form ACTUALLY translated to the top flight:

    gap = (top-flight log strength deviation) / (second-tier log strength deviation)

averaged (median, robust to outliers) over all promoted clubs per league. A gap of
0.7 means a promoted club plays ~70% as far from average, in the top flight, as it
did in the division below.

Writes data-raw/leagues/level_gap_calibration.json and prints the values to paste
into second_tier.LEVEL_GAP.
"""
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import config, history
from leagues.names import canonical, UnknownTeam
from leagues.second_tier import table_strengths

ROOT = Path(__file__).resolve().parents[1]
FEED = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
MIN_DEV = 0.15          # ignore clubs that were ~average in the 2nd tier (ratio unstable)


def _prior_st_code(tf_code: str) -> str:
    """Top-flight season code -> the second-tier code of the season before.
    '2223' (2022-23 top flight) -> '2122' (2021-22 second tier)."""
    return f"{int(tf_code[:2]) - 1:02d}{tf_code[:2]}"


def _tf_strengths(hist: pd.DataFrame, season: str) -> dict:
    """Per-team gf/ga per game + league avg, from actual top-flight results."""
    d = hist[hist["season"] == season]
    agg = {}
    for _, r in d.iterrows():
        h, a, hg, ag = r["home"], r["away"], r["home_goals"], r["away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue
        for t in (h, a):
            agg.setdefault(t, {"gf": 0, "ga": 0, "p": 0})
        agg[h]["gf"] += hg; agg[h]["ga"] += ag; agg[h]["p"] += 1
        agg[a]["gf"] += ag; agg[a]["ga"] += hg; agg[a]["p"] += 1
    lg = sum(t["gf"] for t in agg.values()) / max(sum(t["p"] for t in agg.values()), 1)
    return {t: {"gf_pg": v["gf"] / v["p"], "ga_pg": v["ga"] / v["p"], "lg": lg}
            for t, v in agg.items() if v["p"] > 0}


def _fetch_st(league: str, st_code: str) -> dict:
    code = config.get(league).fd_code2
    try:
        req = urllib.request.Request(FEED.format(season=st_code, code=code),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = table_strengths(io.StringIO(resp.read().decode("utf-8", "replace")), league)
    except Exception as exc:
        print(f"  (no 2nd-tier {code} {st_code}: {exc})")
        return {}
    out = {}
    for team, s in raw.items():
        try:
            out[canonical(team, league)] = s
        except UnknownTeam:
            continue
    return out


def _dev(gf_pg, ga_pg, lg):
    return np.log(max(gf_pg, 0.3) / lg), np.log(max(ga_pg, 0.3) / lg)


def promoted_club_seasons(league: str) -> list:
    """Every (second-tier deviation, top-flight deviation) pair for clubs promoted
    in the last few seasons. Attack and defence kept as separate channels."""
    hist = history.fetch_history(league)
    seasons = sorted(hist["season"].unique())
    rows = []
    for i in range(1, len(seasons)):
        tf, prev = seasons[i], seasons[i - 1]
        tf_str = _tf_strengths(hist, tf)
        prev_tf = set(_tf_strengths(hist, prev))
        promoted = [t for t in tf_str if t not in prev_tf]     # new to the top flight
        if not promoted:
            continue
        st = _fetch_st(league, _prior_st_code(tf))
        for club in promoted:
            if club not in st:
                continue
            st_att, st_def = _dev(st[club]["gf_pg"], st[club]["ga_pg"], st[club]["lg"])
            tf_att, tf_def = _dev(tf_str[club]["gf_pg"], tf_str[club]["ga_pg"], tf_str[club]["lg"])
            rows.append({"league": league, "season": tf, "club": club,
                         "st": [round(st_att, 3), round(st_def, 3)],
                         "tf": [round(tf_att, 3), round(tf_def, 3)]})
    return rows


def _fit(x, y):
    x, y = np.array(x), np.array(y)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return round(float(slope), 3), round(float(intercept), 3), round(r, 2)


def main():
    all_rows = []
    for lg in config.LEAGUES:
        print(f"=== {lg} ===")
        rows = promoted_club_seasons(lg)
        all_rows += rows
        print(f"  {len(rows)} promoted-club seasons resolved")

    # Attack and defence translate very differently, and per-league samples are tiny
    # (~10 clubs), so fit ONE pooled regression per channel across all four leagues.
    sa = [r["st"][0] for r in all_rows]; ta = [r["tf"][0] for r in all_rows]
    sd = [r["st"][1] for r in all_rows]; td = [r["tf"][1] for r in all_rows]
    att = _fit(sa, ta)
    dfn = _fit(sd, td)

    report = {
        "method": "pooled linear fit: top-flight deviation ~ second-tier deviation",
        "n_promoted_club_seasons": len(all_rows),
        "attack": {"slope": att[0], "intercept": att[1], "r": att[2]},
        "defence": {"slope": dfn[0], "intercept": dfn[1], "r": dfn[2],
                    "note": "near-zero r -> slope zeroed in second_tier.DEFENCE_MAP; "
                            "use the intercept (mean) as a constant prior"},
        "detail": all_rows,
    }
    path = ROOT / "data-raw" / "leagues" / "level_gap_calibration.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nATTACK :  tf = {att[0]:+.3f}*st {att[1]:+.3f}   r={att[2]:+.2f}")
    print(f"DEFENCE:  tf = {dfn[0]:+.3f}*st {dfn[1]:+.3f}   r={dfn[2]:+.2f}  (slope zeroed)")
    print("\nsecond_tier.py constants:")
    print(f"    ATTACK_MAP = ({att[0]}, {att[1]})")
    print(f"    DEFENCE_MAP = (0.0, {round(float(np.mean(td)), 2)})")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
