# World Cup 2026 Predictor

Static site (index.html + data/predictions.json) deployed on Render, predicting every WC26 group-stage match. Owner: John (guchhenry91).

## Layout
- `index.html` — UI, reads `data/predictions.json` only. No build step.
- `predict.py` — prediction engine (pure stdlib). Run `python predict.py` to regenerate `data/predictions.json` from `data-raw/`.
- `data-raw/schedule.json` — all 72 group matches (do not change ids).
- `data-raw/ratings.json` — Elo + FIFA per team (baseline; predict.py applies result-based Elo deltas itself — do not manually edit after tournament start).
- `data-raw/news.json` — per-team form, injuries, key players, headlines.
- `data-raw/players.json` — per-team top attacking threats (3–4 each) used for scorer predictions: player, club, pos, club goals/apps, pens.
- `data-raw/results.json` — final scores keyed by match id as STRING.
  - **Group matches (ids 1–72):** `{"5": {"home_goals":2,"away_goals":0}}`.
  - **Knockout matches (ids 73–104:** R32 73–88, R16 89–96, QF 97–100, SF 101–102, 3rd 103, Final 104): record the ACTUAL teams and the advancer, because the projected slotting won't always match FIFA's real draw and pen shootouts must be captured: `{"74": {"home":"Germany","away":"Paraguay","home_goals":1,"away_goals":1,"winner":"Paraguay"}}`. Map an actual result to a bracket id by its HOME team (a group **winner/runner-up**, which is reliable); the projected AWAY team (a third-placed side) may be wrong, so trust the web result for the real opponent. `winner` = the team that advanced (crucial when a tie is level after extra time and decided on penalties). `knockout.py` then displays the real teams and grades the model's pick against `winner`.
- `data-raw/bracket.json` — knockout bracket structure (slots, dates, venues). Static; do not edit.
- `predict.py` runs a 20k-sim Monte Carlo each time → `knockout` section in predictions.json (title odds + projected bracket). Set env `WC_SIMS` to change sim count.
- `data-raw/picks_log.json` — **auto-managed pick tracker** (do not hand-edit). predict.py locks each match's pick before kickoff and freezes it once a result lands, so the win/loss record grades the genuine pre-match pick (never a hindsight re-computation). It is committed each run so the record persists. `record` in predictions.json = `{correct, wrong, total, pending, by_confidence}`.

## Publishing (IMPORTANT)
Never run predict.py + git + deploy hook by hand in the tasks. After editing any data file, run **`python deploy.py "<message>"`** — one atomic step that re-runs the model, grades locked picks, commits, pushes, and triggers the Render deploy. It self-heals a stale publish (re-runs the model so unpublished results get caught) and always re-triggers the deploy, so the live site can never lag the repo. `status.py` is the gate helper: prints `finished_unrecorded` + `upcoming_4h` so a task knows whether there's anything to do.

## Daily update procedure (run every morning)
1. Web-search final scores of all WC matches played yesterday (and any missed earlier); add them to `data-raw/results.json`.
2. Web-search overnight team news: injuries, suspensions, lineup news for teams playing TODAY and TOMORROW. Update those teams' entries in `data-raw/news.json` (update `form` strings with yesterday's results too, and set top-level `updated`).
3. If any player listed in `data-raw/players.json` for a team playing today/tomorrow is newly ruled OUT, remove them (so scorer picks stay accurate).
4. Run `python predict.py`. Verify it prints "Wrote 72 matches" and no errors.
5. Commit all changes ("daily update YYYY-MM-DD: results + news") and push to main.
6. Render is connected by public Git URL and does NOT auto-deploy on push — trigger a redeploy by POSTing the deploy hook stored at `C:\Users\John\.claude\worldcup-deploy-hook.txt` (PowerShell: `Invoke-RestMethod -Method Post -Uri "<hook>"`).

Rules: never invent scores or injuries — only verified info. Team names must exactly match the names in schedule.json. Keep reasons concise.
