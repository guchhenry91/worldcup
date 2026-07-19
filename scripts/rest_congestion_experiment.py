"""Does rest/congestion carry signal the model does not already have?

The model sees only who played whom and the score. It has no idea a side played
72 hours ago and its opponent had a week. If tired teams underperform the model's
expectation, that is information the model lacks -- and unlike the market, it is
free and already sitting in our own fixture list.

Test: model's expected home points vs ACTUAL home points, split by rest advantage.
If rest carries nothing, residuals are flat across the buckets.
"""
import json, numpy as np, pandas as pd
from leagues import backtest, dataset

P = json.load(open('data-raw/leagues/backtest_report.json'))
out = []
for lg in ("PL","LALIGA","BUNDESLIGA","LIGUE1"):
    m = dataset.build_matches(lg).copy()
    m["date"] = pd.to_datetime(m["date"])
    # days since each side's previous fixture
    last = {}
    rest_h, rest_a = [], []
    for _, r in m.sort_values("date").iterrows():
        for team, store in ((r["home"], rest_h), (r["away"], rest_a)):
            prev = last.get(team)
            store.append(np.nan if prev is None else (r["date"] - prev).days)
        last[r["home"]] = last[r["away"]] = r["date"]
    ms = m.sort_values("date").copy()
    ms["rest_h"], ms["rest_a"] = rest_h, rest_a

    p = P[lg]
    res = backtest.walk_forward(m, xi=p["xi"], xg_weight=p["xg_weight"])
    res = res.merge(ms[["date","home","away","rest_h","rest_a"]],
                    on=["date","home","away"], how="left")
    out.append(res)

d = pd.concat(out, ignore_index=True).dropna(subset=["rest_h","rest_a"])
d = d[(d.rest_h.between(2,14)) & (d.rest_a.between(2,14))]
y = d["y"].to_numpy() if "y" in d.columns else d["outcome"].to_numpy()
# expected home points from the model vs actual
exp_pts = 3*d["p_home"].to_numpy() + 1*d["p_draw"].to_numpy()
act_pts = np.where(y == 0, 3.0, np.where(y == 1, 1.0, 0.0))
d = d.assign(resid=act_pts - exp_pts, diff=d.rest_h - d.rest_a)

print(f"n = {len(d)} matches with known rest for both sides\n")
print("  home rest advantage      n      model expected   actual    residual")
for lo, hi, lab in ((-99,-3,"much less rested (<=-3d)"),(-3,-1,"less rested (-2d)"),
                    (-1,2,"level (-1 to +1)"),(2,4,"more rested (+2/3d)"),
                    (4,99,"much more rested (>=+4d)")):
    s = d[(d["diff"]>=lo)&(d["diff"]<hi)]
    if len(s) < 100: continue
    se = s["resid"].std()/np.sqrt(len(s))
    sig = "  <-- signal" if abs(s["resid"].mean()) > 1.96*se else ""
    print(f"  {lab:<24} {len(s):<6} {(3*s.p_home+s.p_draw).mean():.3f}          "
          f"{(s.resid + 3*s.p_home + s.p_draw).mean():.3f}    "
          f"{s['resid'].mean():+.3f} +/-{1.96*se:.3f}{sig}")
