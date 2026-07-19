"""Does a COMPLETED season's cached data still match what Understat serves?

The whole argument for caching history is that finished seasons cannot change.
If Understat has revised its xG model or backfilled corrections, that argument
fails and a periodic full refresh is worth having.
"""
import pandas as pd, soccerdata as sd

LG, SEASON = "ENG-Premier League", "2223"      # long finished

cached = sd.Understat(leagues=LG, seasons=SEASON).read_team_match_stats().reset_index()
fresh  = sd.Understat(leagues=LG, seasons=SEASON, no_cache=True).read_team_match_stats().reset_index()

print(f"cached rows {len(cached)}   fresh rows {len(fresh)}")
key = [c for c in ("game", "team") if c in cached.columns]
num = [c for c in cached.columns if pd.api.types.is_numeric_dtype(cached[c])]
c = cached.set_index(key)[num].sort_index()
f = fresh.set_index(key)[num].sort_index()
if not c.index.equals(f.index):
    print("INDEX DIFFERS -> content changed")
else:
    d = (c - f).abs()
    worst = d.max().sort_values(ascending=False)
    print("\nlargest absolute difference per column:")
    print(worst.head(8).to_string())
    total = float(d.to_numpy().sum())
    print(f"\ntotal absolute difference across every numeric cell: {total:.6f}")
    print("VERDICT:", "IDENTICAL - cache is safe" if total < 1e-9
          else "CHANGED - upstream revised history, periodic refresh IS worth it")
