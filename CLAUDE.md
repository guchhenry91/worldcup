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

---

# Leagues engine (Premier League — Phase 2a)

A second, independent predictor living in `leagues/`. It shares the repo and
`deploy.py` with the World Cup app but **touches none of its files**: the WC
engine stays pure-stdlib, the league engine needs pandas/scipy/penaltyblog.

- `leagues.html` + `leagues.css` — the PL page. Reads `data/leagues/pl.json` and
  `data/leagues/clubs.json` only. Linked to/from `index.html` by a plain switcher.
- `python -m leagues.publish` — the one command. Fits the model, sims the season,
  builds player props, locks picks, writes `data/leagues/pl.json` atomically.
- `python -m leagues.tune` — the match-model gate (walk-forward vs de-vigged
  closing odds). `python -m leagues.props_backtest` — the props gate.

## Data sources
- Results + closing odds: football-data.co.uk. Team xG: Understat.
- **Players: Understat season stats + shot events** — NOT FBref. soccerdata's
  FBref player-match reader drives a headless Chrome per match page (~4/min):
  five seasons of one league is ~8 hours. Understat gives the same signal in
  seconds.
- **Penalties are unlabelled**: soccerdata maps Understat's "Penalty" situation
  to NA. Match on NA, not on the string, or every club silently gets no penalty
  taker (see the regression test in `tests/leagues/test_players.py`).
- **Promoted-club priors come from second-tier form**, NOT ClubElo (which was a
  single third-party point of failure — down for days). Each promoted club's prior
  is derived from its actual second-division season (football-data.co.uk E1/SP2/D2/F2)
  via a calibrated linear map: attack carries a mild signal, defence none (promoted
  clubs concede ~+0.19 above average regardless). See `leagues/second_tier.py` and
  `scripts/calibrate_level_gap.py`. A club that can't be resolved in the second-tier
  feed falls back to the weakest-side seed with a `data_warnings` note.
- **soccerdata shot-events bug**: `read_shot_events` crashes on GER-Bundesliga (a
  match roster returns as a list, not a dict). Both `fetch_player_logs` and
  `team_shot_context` guard it and degrade (SOT via league-average ratio, neutral
  opponent factors) rather than sink the league.

## Scheduled jobs — NOT YET REGISTERED
`ops/leagues_weekly.py` and `ops/leagues_matchday.py` are written and working but
deliberately unregistered: the 2026-27 seasons start **2026-08-21** (PL MW1: Arsenal
v Coventry; the other three the same weekend), and before then they would republish
unchanged files every week. Both call `publish.main()`, which loops **all four
leagues** (PL, La Liga, Bundesliga, Ligue 1), aborting per-league on failure, then
`deploy.py`.

**Register them in mid-August:**
- `leagues-weekly` — cron `0 6 * * 2` (Tue 06:00, after Monday-night football).
- `leagues-matchday` — cron `0 23 * * 6,0` (Sat/Sun 23:00, after the day's games).

Both abort rather than deploy if a fetch fails — never ship a stale-but-fresh-
looking file.
