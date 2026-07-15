# Leagues Phase 2b — Four Leagues + Unified Competition Switcher

**Date:** 2026-07-14
**Parent spec:** `docs/superpowers/specs/2026-07-13-multileague-predictor-design.md`
**Predecessor:** `docs/superpowers/specs/2026-07-14-leagues-phase2a-pl-slice-design.md` (Premier League vertical slice — built, gates passed)

---

## 1. Goal

Generalise the proven Premier League slice to **all four leagues** (PL, La Liga, Bundesliga, Ligue 1), and unify the World Cup app and the four league pages into **one switcher-driven site**. This closes out the original multi-league spec.

Phase 2a proved the contract on one league. Phase 2b is mostly a loop plus a UI merge — the model, props, sim, picks and orchestrator are already league-generic (`build(league)` takes a key; `names.ALIASES` already covers all four leagues from Phase 1).

---

## 2. Scope

**In:**
- Publish La Liga, Bundesliga, Ligue 1 alongside the PL (`data/leagues/{laliga,bundesliga,ligue1}.json`).
- Per-league club colour/crest maps (~60 more clubs).
- Confirm each league's match gate and props gate; **honestly flag any that underperform** rather than shipping them silently.
- A single `index.html` with a top-level competition switcher: **World Cup · Premier League · La Liga · Bundesliga · Ligue 1**, each loading its own data file, reusing one shared set of card/table/performance components.
- Backfill the "Model Performance" panel onto the World Cup view (spec build-phase 8).
- **Live model-vs-market edge display** on upcoming league fixtures: each card shows the bookmaker's current de-vigged 1X2 next to the model's probabilities and the disagreement. Source: football-data.co.uk `fixtures.csv` — the SAME source as the historical backtest, so `devig()` and the `names.py` mapping are reused unchanged. Off-season caveat: `fixtures.csv` only lists ~1 week ahead, so real top-flight odds do not appear until ~mid-August; the loader is unit-testable now against the stable CSV format but not live-verifiable until then.
- Extend the ops jobs to refresh all four leagues.

**Out (later phases / explicitly deferred):**
- Champions League / domestic cups (deferred in the parent spec until their draws).
- Second-tier data to fix promoted-club player props (structural gap, not a 2b task).
- Any paid data source.

**Hard constraints:**
- **The unified `index.html` must not ship until after the World Cup final (2026-07-19).** Until then the WC app stays exactly as it is on `main`. The new page is built and verified behind a copy (`app.html` in the worktree), and only becomes `index.html` once the WC winds down.
- **ClubElo must be reachable at build time.** Every league's promoted-club priors depend on it. A league built during a ClubElo outage carries the same degraded-prior warning as the PL does now; we do not publish a league's final table on a fallback prior without the on-page warning.

---

## 3. Gate (blocks the UI merge)

Before any league appears in the switcher, it must clear the same two gates the PL did:

- **Match gate** (`leagues.tune`): walk-forward vs de-vigged closing odds, RPS gap in +0.005…+0.02, accuracy 50-55%. Already passing for all four (PL +0.0061, La Liga +0.0055, Bundesliga +0.0066, Ligue 1 +0.0058) — re-confirm under the current model.
- **Props gate** (`leagues.props_backtest`): scorer MAE and shots MAE beat the position-average baseline. Proven for PL; must be run per league.

**Bundesliga watch item:** its match gate tuned to `xg_weight=1.0` (pure xG, ignoring realised goals). Plausible for a high-variance league, but re-confirm it is not masking a data-join problem (xG coverage, name mismatches) before shipping.

**Honesty rule (from the parent spec):** if a league fails a gate, it does not go in the switcher. We say so, rather than ship vibes.

---

## 4. Architecture

Almost everything already exists and is league-generic. The deltas:

| Area | Phase 2a state | Phase 2b change |
|---|---|---|
| `leagues/publish.py` | `build(league)` generic; `main()` hardcodes PL | `main()` loops the four leagues, writes one JSON each, atomically |
| `data/leagues/clubs.json` | 20 PL clubs | Per-league colour maps for all four (one file per league, or one keyed by league) |
| `leagues/names.py` | 4 leagues, ~29 clubs each | Fill gaps found by the fixture cross-check per league |
| UI | `leagues.html` (PL only) + separate `index.html` (WC) | One `app.html` with a switcher and a data-driven view layer; becomes `index.html` post-final |
| WC performance | none | performance panel reads the WC backtest, shown on the WC view |
| ops | PL only | loop four leagues; abort-per-league on fetch failure |

**Data-driven view layer.** The switcher is the only new UI concept. Each competition is described by a small manifest: `{key, label, dataUrl, kind: "cup" | "league"}`. Selecting one fetches its data file and renders it through shared components. Cup (WC) and league views differ only in which sections they show (a cup has a bracket + group stage; a league has a projected table + relegation). The card, table, props list, and performance panel are identical.

**Why one page, not five.** The parent spec's end state, and it removes the CSS duplication that Phase 2a accepted as a temporary cost. The risk — editing the live WC app — is bounded by building it as `app.html` first and swapping only after the final.

---

## 5. The WC data contract

The WC app currently emits `data/predictions.json` in a shape the switcher must also read. Two options, decided here: **the switcher adapts to both shapes rather than rewriting the WC engine.** `predict.py` stays pure-stdlib and untouched; the view layer has a small adapter that maps the WC payload and the league payload onto the same component props. Rewriting `predict.py`'s output would risk the live app for no user-visible gain.

The only WC-side addition is the **performance panel**: it needs the WC backtest numbers. If `predict.py` does not already emit them, they are computed once and written into `predictions.json` under a `backtest` key by a small, isolated addition — TDD'd, and verified not to change any existing field.

---

## 6. Build phases (for the implementation plan)

1. **Per-league name-map + colour completeness** — cross-check every fixture team against `names.ALIASES` and the colour map for each league; fill gaps; fail loudly on any unmapped club.
2. **Generalise `publish.main()`** — loop the four leagues, write four JSONs atomically, one abort-per-league.
3. **Per-league gates** — run `tune` and `props_backtest` for all four; record reports; confirm or flag each.
4. **`app.html` switcher + shared view layer** — build the unified page as a copy; WC + 4 leagues selectable; verify each renders from real data.
5. **WC performance panel + backtest emission** — isolated WC-side addition, TDD'd, no existing field changed.
6. **Swap `index.html`** — after the WC final only; the old file preserved in git history; verify the WC view is unchanged from the user's point of view.
7. **Ops** — extend the weekly/match-day jobs to loop four leagues.

---

## 7. Risks

- **Editing the live WC app.** Mitigated by building behind `app.html` and swapping only post-final; the swap is one file rename verified against a screenshot of the current live app.
- **ClubElo outage at build time** (currently live). Mitigated by the existing loud-fail + on-page warning; a league is not shipped with a silently-degraded table.
- **Bundesliga `xg_weight=1.0`.** Re-confirm it is signal, not a join bug, before shipping (§3).
- **Cross-league name collisions** — e.g. two clubs sharing a short name across leagues. `names.ALIASES` is already keyed by league, so this is contained; the fixture cross-check in build-phase 1 is the safety net.
- **Scope creep** — CL/cups stay deferred; promoted-club player props stay a known gap, not a 2b fix.
