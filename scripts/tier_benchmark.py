"""Does the p>=0.65 tier actually beat the market on the SAME fixtures?

The 77.4% billing has never been compared with what the market would have
achieved on the identical subset. Picking only strong favourites hits ~77%
almost by construction, so the number alone cannot separate skill from
selection. This settles it.
"""
import numpy as np, pandas as pd, json
from leagues import backtest, dataset, config

PARAMS = json.load(open('data-raw/leagues/backtest_report.json'))
rows = []
for lg in ("PL", "LALIGA", "BUNDESLIGA", "LIGUE1"):
    p = PARAMS[lg]
    r = backtest.walk_forward(dataset.build_matches(lg),
                              xi=p["xi"], xg_weight=p["xg_weight"])
    r = r[r.get("m_home").notna()] if "m_home" in r.columns else r.iloc[0:0]
    if r.empty:
        continue
    r["league"] = lg
    rows.append(r)

d = pd.concat(rows, ignore_index=True)
mp = d[["p_home", "p_draw", "p_away"]].to_numpy()
kp = d[["m_home", "m_draw", "m_away"]].to_numpy()
y = d["y"].to_numpy() if "y" in d.columns else d["outcome"].to_numpy()

model_pick = mp.argmax(1)
mkt_pick = kp.argmax(1)
model_p = mp.max(1)

print(f"pooled n = {len(d)}")
for thr in (0.55, 0.60, 0.65, 0.70):
    sel = model_p >= thr
    n = int(sel.sum())
    if not n:
        continue
    m_hit = (model_pick[sel] == y[sel]).mean()
    k_hit = (mkt_pick[sel] == y[sel]).mean()
    # how often the market's own pick agrees with ours on this subset
    agree = (model_pick[sel] == mkt_pick[sel]).mean()
    se = np.sqrt(m_hit * (1 - m_hit) / n)
    print(f"p>={thr:.2f}  n={n:<5} model {m_hit:6.1%} (+/-{1.96*se:.1%})   "
          f"market {k_hit:6.1%}   diff {m_hit-k_hit:+.1%}   agree {agree:.0%}")
