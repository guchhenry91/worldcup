"""Is our Over/Under 2.5 call actually good, or just good-looking?

57.9% correct sounds strong against a 50% coin flip, but O/U 2.5 is a liquid,
sharply-priced market -- the honest question is how we do against the DE-VIGGED
closing line on the same matches, exactly as we benchmark 1X2. Accuracy alone can
flatter a model that simply follows the base rate (most matches go over).

Reports accuracy, log-loss and Brier for model vs market, plus the base-rate
baseline (always predict the more common outcome), so a number only goes on the
page if it survives all three comparisons.
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import config, dataset
from leagues.model import LeagueModel, scoreline_grid, goals_markets
from leagues.names import canonical, UnknownTeam

ROOT = Path(__file__).resolve().parents[1]
FEED = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def market_ou(league="PL"):
    """(date, home, away) -> de-vigged P(over 2.5) from the market average."""
    lg = config.get(league)
    out = {}
    for season in lg.history_seasons:
        try:
            req = urllib.request.Request(FEED.format(season=season, code=lg.fd_code),
                                         headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
            d = pd.read_csv(io.StringIO(raw))
        except Exception as exc:
            print(f"  skip {season}: {exc}")
            continue
        if "Avg>2.5" not in d.columns:
            continue
        for _, r in d.iterrows():
            o, u = r.get("Avg>2.5"), r.get("Avg<2.5")
            if pd.isna(o) or pd.isna(u):
                continue
            try:
                h = canonical(r["HomeTeam"], league)
                a = canonical(r["AwayTeam"], league)
            except (UnknownTeam, KeyError):
                continue
            io_, iu = 1.0 / float(o), 1.0 / float(u)
            out[(pd.to_datetime(r["Date"], dayfirst=True).normalize(), h, a)] = io_ / (io_ + iu)
    return out


def main(league="PL", min_train=760, step_days=7):
    mkt = market_ou(league)
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        tr = df[df["date"] < cutoff]
        te = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if tr.empty or te.empty:
            continue
        try:
            mod = LeagueModel().fit(tr, ref=cutoff)
        except Exception:
            continue
        for _, m in te.iterrows():
            try:
                lh, la = mod.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            g = scoreline_grid(lh, la, mod.rho)
            mk = goals_markets(g)
            key = (pd.Timestamp(m["date"]).normalize(), m["home"], m["away"])
            rows.append({"p_model": mk["p_over25"], "p_market": mkt.get(key),
                         "over": int(m["home_goals"]) + int(m["away_goals"]) > 2.5})
    r = pd.DataFrame(rows)
    both = r.dropna(subset=["p_market"])

    def stats(p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
        y = np.asarray(y, dtype=float)
        return {"acc": round(100 * float(((p > 0.5) == (y > 0.5)).mean()), 1),
                "logloss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
                "brier": round(float(((p - y) ** 2).mean()), 4)}

    base = float(both["over"].mean())          # always-predict-the-common-outcome
    out = {
        "league": league, "n_all": int(len(r)), "n_with_market": int(len(both)),
        "over_base_rate_pct": round(100 * base, 1),
        "model": stats(both["p_model"], both["over"]),
        "market": stats(both["p_market"], both["over"]),
        "always_majority": {"acc": round(100 * max(base, 1 - base), 1)},
    }
    print(json.dumps(out, indent=2))
    (ROOT / "data-raw" / "leagues" / "goals_market_benchmark.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
