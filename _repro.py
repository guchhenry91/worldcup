import json, sys, types
from pathlib import Path
import pandas as pd
import leagues.publish as publish
from leagues import picks, fixtures, config

OUT = Path("_scratch_out"); OUT.mkdir(exist_ok=True)
PK  = Path("_scratch_picks"); (PK/"pl").mkdir(parents=True, exist_ok=True)
publish.OUT = OUT
publish.PICKS_DIR = PK

# one PL fixture, unplayed, kicking off soon; locked pick p=0.72 in the log
KO = "2026-08-21T19:00:00+00:00"
log = {"2026:1": {"pick":"Arsenal","confidence":5,"p_pick":0.72,
                  "locked_at":KO,"kickoff":KO,"tainted":False}}
picks.save_log(log, PK/"pl"/"picks_log.json")

payload = {"matches":[{"id":1,"matchweek":1,"date":KO,"home":"Arsenal","away":"Coventry City",
    "prediction":{"pick":"Arsenal","confidence":5,"p_pick":0.72,"best_pick":True,
                  "provisional":False,"score":"2-0"}}]}
(OUT/"pl.json").write_text(json.dumps(payload), encoding="utf-8")
for f in ("laliga.json","bundesliga.json","ligue1.json"):
    (OUT/f).write_text(json.dumps({"matches":[]}), encoding="utf-8")

FX = pd.DataFrame([{"match_id":1,"round":1,"date":pd.Timestamp(KO),"venue":"",
                    "home":"Arsenal","away":"Coventry City","home_goals":pd.NA,
                    "away_goals":pd.NA,"played":False}])
publish.fixtures = types.SimpleNamespace(fetch_fixtures=lambda lg: FX if lg=="PL" else FX.iloc[0:0])

best = publish.build_best_picks()
print("upcoming entries:", len(best["upcoming"]))
for u in best["upcoming"]:
    print("  ", u["league"], u["id"], u["pick"], u["p_pick"], "score=", u.get("score"), "prov=", u.get("provisional"))
