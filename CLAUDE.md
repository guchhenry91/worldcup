# Henry's Match Engine

Static site deployed on Render. The live product (index.html) is a four-league
football predictor -- Premier League, La Liga, Bundesliga, Ligue 1 -- plus two
cross-league boards (Best Picks, Player Picks) and a Grades tab. Owner: John
(guchhenry91).

Repo also holds `worldcup.html`, the completed World Cup 2026 tournament
(daily picks, group projections, title odds, full graded knockout bracket).
It is an ARCHIVE, kept exactly as it finished -- final record 64-26 of 90
(71%), champion Spain -- and should not need to change again.

## Layout
- `index.html` — the SITE ENTRY POINT: the unified predictor (Best Picks, Player
  Picks, Grades, and the four leagues). Reads `data/leagues/*.json`. No build step.
- `worldcup.html` — the completed World Cup 2026 tournament, preserved verbatim:
  daily picks, group projections, title odds and the full graded knockout bracket.
  Reads `data/predictions.json` only. Final record **64-26 of 90 (71%)**, champion
  Spain. Linked from the switcher; it links back. It is an ARCHIVE -- the tournament
  is over, so this page should not need to change again.
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

# Leagues engine (the live product: PL, La Liga, Bundesliga, Ligue 1)

The predictor living in `leagues/`. It shares the repo and `deploy.py` with
the archived World Cup app but **touches none of its files**: the WC
engine stays pure-stdlib, the league engine needs pandas/scipy/penaltyblog.

- `index.html` + `app.css` — the unified UI for all four leagues plus the two
  cross-league boards. (`leagues.html`/`leagues.css` were the old single-league
  page and were removed when `app.html` became `index.html`.)
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

## The three published boards
- `data/leagues/best.json` — cross-league **match-winner** picks at p>=0.65.
- `data/leagues/player_picks.json` — cross-league **player** picks in three markets:
  anytime goalscorer, 2+ shot attempts, 1+ shot on target. Bars are PER MARKET
  (`PLAYER_PICK_MIN_PROB`): shots/SOT 0.70, goalscorer **0.40**. The goalscorer bar
  is lower because it must be — a team scores ~1.5 goals and one man takes a share,
  so the best anytime price in any of these leagues is ~50%. A 0.70 bar there would
  publish an empty section forever, not a stricter one.
- Both are graded **separately**, and player picks are also graded per market: a 45%
  goalscorer and an 80% shots pick are both near their market's ceiling, so pooling
  them yields a headline number describing neither. The `grades` tab shows all tiers.

**Player picks are graded from shot events**, not from `fetch_player_logs` (which is
one row per player-SEASON and cannot say whether a man scored in a given fixture).
`players.match_player_stats()` counts goals/shots/SOT per player per game, INCLUDING
penalties (an anytime pick wins on a penalty; `np_goals` would grade that a miss) and
EXCLUDING own goals. A player with no shot row grades **wrong, not void** — the feed
cannot separate "didn't play" from "played, never shot", so we take the harsher
reading deliberately: it can only understate the record, never inflate it.

**Bundesliga cannot be graded** (its shot events crash upstream), so its player picks
publish with `gradeable: false`, are excluded from the record rather than parked as
permanent "pending", and the SOT market is withheld there entirely — without a shot
feed the on-target ratio is a league average, i.e. an assumption, not a measurement.

**`MIN_SQUAD_FOR_PROPS = 6`** — a team with fewer players in the rates table gets NO
props. The sigma-lambda rescale forces a team's players to sum to the match lambda,
so a near-empty squad hands one man the whole team's goals: promoted Schalke had a
single player with top-flight history and published as a **72.8% anytime scorer**
when nothing else in four leagues beat 50.8%. One player is more dangerous than
none — none is visibly a hole, one looks like the best pick on the board.

## Roster verification
`python -m scripts.sync_rosters` snapshots every current 2026-27 club and player
from the ESPN league/team roster feeds into `data-raw/leagues/rosters.json`.
`python -m scripts.roster_integrity_check` verifies league membership, duplicate
player IDs and visibly reports thin/incomplete source rosters. The snapshot is
dated and provisional while the summer registration window remains open. It is
an identity/eligibility source only; Understat remains the performance-rate
source, so a player with no usable history is never assigned invented scoring or
shot rates. The snapshot CORROBORATES, it does not convict. Where a club's roster is
COMPLETE (>=18 listed) it is authoritative: it reassigns a player's club, and a
player absent from it is dropped as departed. Where a roster is THIN, MISSING or
STALE we keep the existing attribution and warn on the page, because absence from
incomplete evidence is not evidence of absence.

That distinction was learned the hard way. Treating thin rosters as proof deleted
Real Madrid, Barcelona, Atletico, PSG, Marseille and 14 of 18 Bundesliga clubs --
Mbappe and Raphinha among them, 70% of La Liga and Ligue 1 -- because the free feed
happened to list fewer than 18 names for them. A surname rescue also runs, but only
within the club a player is ALREADY at, so it can never invent a transfer; without
it Understat's "Thiago" vs the feed's "Igor Thiago" deleted a real Brentford player
over a spelling difference. The match model is unaffected either way.

## Scheduled jobs
`ops/leagues_weekly.py` and `ops/leagues_matchday.py` are manual wrappers around
the four-league publish path. They pass `--league-data` to `deploy.py`, so a league
refresh never regenerates or stages World Cup files.

**DO NOT REGISTER THEM.** This instruction is superseded and following it would
cause an incident. `.github/workflows/leagues.yml` already runs the same job on the
same crons (`0 6 * * 2`, `0 23 * * 6,0`). Registering these locally too means two
processes publishing, both committing to the same repo, both triggering Render —
git conflicts and double deploys, on a schedule.

GitHub Actions wins because it does not need the laptop on or the app open, which
was the whole point of moving there. Keep `ops/leagues_weekly.py` and
`ops/leagues_matchday.py` as MANUAL commands (`python -m ops.leagues_weekly` for a
publish-and-deploy on demand) — just never on a schedule.

The one scheduled task that IS correct to have locally is `leagues-matchday-news`
(Fri/Sat mornings): it needs judgement about confirmed XI vs rumour, which is why
it cannot live in Actions.

Both abort rather than deploy if a fetch fails — never ship a stale-but-fresh-
looking file.
