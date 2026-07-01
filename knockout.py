"""Monte Carlo knockout simulation for the 2026 World Cup (48-team format).

Given current ratings and any played results, this simulates the rest of the
group stage thousands of times to estimate every team's odds of reaching each
round and winning the cup, and builds a single deterministic "projected
bracket" from the expected group standings.

Self-contained (stdlib only). predict.py passes in the data and a couple of
shared callables so probabilities stay consistent with the match cards.
"""
import math
import random


def _poisson(lam, rng):
    """Knuth sampler — fine for the small lambdas (~0.2-3) we use."""
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _ko_winprob(dr):
    """No-draw knockout win probability for the higher-rated (home) side."""
    return 1.0 / (1.0 + 10 ** (-dr / 400.0))


def _assign_thirds(qual_groups, slots):
    """Assign the qualifying third-placed groups to third-slots respecting
    each slot's allowed-group list. Returns {slot_index: group} or {} on fail.

    qual_groups: list of group letters whose third-placed team qualified (<=8).
    slots: list of (slot_index, allowed_set).
    """
    order = sorted(range(len(slots)), key=lambda i: len(slots[i][1]))
    result = {}
    used = set()

    def bt(pos):
        if pos == len(order):
            return True
        si = order[pos]
        _, allowed = slots[si]
        for g in qual_groups:
            if g not in used and g in allowed:
                used.add(g)
                result[si] = g
                if bt(pos + 1):
                    return True
                used.discard(g)
                del result[si]
        return False

    if bt(0):
        return result
    # fallback: greedy fill (keeps sim robust on any rare unmatched subset)
    res, used = {}, set()
    for si, allowed in slots:
        for g in qual_groups:
            if g not in used and g in allowed:
                used.add(g); res[si] = g; break
    leftover = [g for g in qual_groups if g not in used]
    for si, _ in slots:
        if si not in res and leftover:
            res[si] = leftover.pop()
    return res


class Bracket:
    """Resolves the R32->Final structure once teams are known."""

    def __init__(self, bracket_json, match_country, home_adv):
        self.rounds = bracket_json["rounds"]
        self.third_place = bracket_json.get("third_place")
        self.match_country = match_country
        self.home_adv = home_adv
        # flat id -> match meta
        self.meta = {}
        for rd in self.rounds:
            for m in rd["matches"]:
                self.meta[m["id"]] = (m, rd["name"])
        if self.third_place:
            self.meta[self.third_place["id"]] = (self.third_place, "Third place")
        # the eight third-placed slots (match id, allowed group set)
        self.third_slots = []
        for m in self.rounds[0]["matches"]:
            for side in ("home", "away"):
                if m[side]["t"] == "3":
                    self.third_slots.append((m["id"], side, set(m[side]["g"])))

    def adj(self, team, city, elo):
        return elo.get(team, 1500) + (self.home_adv if self.match_country(city) == team else 0)


def run(elo, groups, gmatches, standings, bracket_json, results,
        match_country, home_adv, elo_per_goal, total_goals, max_sup, sims=20000):
    """Return knockout odds + a projected bracket dict for predictions.json."""
    rng = random.Random(20260611)
    br = Bracket(bracket_json, match_country, home_adv)
    r32 = br.rounds[0]["matches"]

    # pre-split group matches: played vs to-simulate (with lambdas)
    gm_by_group = {g: [] for g in groups}
    for m in gmatches:
        gm_by_group[m["group"]].append(m)

    teams = [t for ts in groups.values() for t in ts]
    counts = {t: dict(r32=0, r16=0, qf=0, sf=0, final=0, champ=0) for t in teams}

    def sim_group(gm_list, members):
        tab = {t: [0, 0, 0] for t in members}  # pts, gd, gf
        for m in gm_list:
            if m.get("result"):
                gh, ga = m["result"]["home_goals"], m["result"]["away_goals"]
            else:
                gh, ga = _poisson(m["lh"], rng), _poisson(m["la"], rng)
            h, a = m["home"], m["away"]
            tab[h][1] += gh - ga; tab[h][2] += gh
            tab[a][1] += ga - gh; tab[a][2] += ga
            if gh > ga: tab[h][0] += 3
            elif ga > gh: tab[a][0] += 3
            else: tab[h][0] += 1; tab[a][0] += 1
        ranked = sorted(members, key=lambda t: (-tab[t][0], -tab[t][1], -tab[t][2], rng.random()))
        return ranked, tab

    def winner(home, away, city):
        dr = br.adj(home, city, elo) - br.adj(away, city, elo)
        return home if rng.random() < _ko_winprob(dr) else away

    for _ in range(sims):
        winners, runners, thirds = {}, {}, []
        for g, members in groups.items():
            ranked, tab = sim_group(gm_by_group[g], members)
            winners[g], runners[g] = ranked[0], ranked[1]
            t3 = ranked[2]
            thirds.append((g, t3, tab[t3][0], tab[t3][1], tab[t3][2]))
        thirds.sort(key=lambda x: (-x[2], -x[3], -x[4], rng.random()))
        qual = thirds[:8]
        qgroups = [g for g, *_ in qual]
        third_team = {g: t for g, t, *_ in qual}
        slot_assign = _assign_thirds(qgroups, [(i, s) for i, (_, _, s) in enumerate(br.third_slots)])
        slot_group = {br.third_slots[i][0]: g for i, g in slot_assign.items()}

        # qualified teams
        for g in groups:
            counts[winners[g]]["r32"] += 1
            counts[runners[g]]["r32"] += 1
        for g, t, *_ in qual:
            counts[t]["r32"] += 1

        res = {}  # match id -> winner team
        # Round of 32
        for m in r32:
            home = winners[m["home"]["g"]] if m["home"]["t"] == "W" else (
                   runners[m["home"]["g"]] if m["home"]["t"] == "R" else third_team.get(slot_group.get(m["id"])))
            away = winners[m["away"]["g"]] if m["away"]["t"] == "W" else (
                   runners[m["away"]["g"]] if m["away"]["t"] == "R" else third_team.get(slot_group.get(m["id"])))
            if home is None or away is None:
                # unmatched third fallback: skip cleanly
                res[m["id"]] = home or away
                continue
            res[m["id"]] = winner(home, away, m["city"])
            counts[res[m["id"]]]["r16"] += 1
        # later rounds
        round_key = {"Round of 16": "qf", "Quarter-finals": "sf", "Semi-finals": "final", "Final": "champ"}
        for rd in br.rounds[1:]:
            ck = round_key[rd["name"]]
            for m in rd["matches"]:
                home = res.get(m["home"]["m"]); away = res.get(m["away"]["m"])
                if home is None or away is None:
                    res[m["id"]] = home or away; continue
                res[m["id"]] = winner(home, away, m["city"])
                counts[res[m["id"]]][ck] += 1

    odds = []
    for t in teams:
        c = counts[t]
        odds.append({
            "team": t,
            "qualify": round(c["r32"] / sims, 3),
            "r16": round(c["r16"] / sims, 3),
            "qf": round(c["qf"] / sims, 3),
            "sf": round(c["sf"] / sims, 3),
            "final": round(c["final"] / sims, 3),
            "champion": round(c["champ"] / sims, 3),
        })
    odds.sort(key=lambda x: (-x["champion"], -x["final"], -x["sf"]))

    bracket = _project(elo, groups, standings, br, r32, results, match_country, home_adv,
                       elo_per_goal, total_goals, max_sup)
    return {
        "sims": sims,
        "projected_champion": bracket["champion"],
        "odds": odds,
        "bracket": bracket["matches"],
    }


def _best_score(dr, elo_per_goal, total_goals, max_sup):
    """Representative scoreline for the favoured (positive-dr) side. Knockout
    ties can't end level, so a dead-heat is nudged 1 goal to the favourite."""
    sup = max(-max_sup, min(max_sup, dr / elo_per_goal))
    gh = round(max(0.2, (total_goals + sup) / 2.0))
    ga = round(max(0.2, (total_goals - sup) / 2.0))
    if gh <= ga:
        gh = ga + 1
    return f"{gh}-{ga}"


def _project(elo, groups, standings, br, r32, results, match_country, home_adv,
             elo_per_goal, total_goals, max_sup):
    """Deterministic bracket from expected standings; honours real results."""
    winners = {g: standings[g][0]["team"] for g in groups}
    runners = {g: standings[g][1]["team"] for g in groups}
    thirds_all = sorted(
        [(g, standings[g][2]["team"], standings[g][2]["exp_pts"]) for g in groups],
        key=lambda x: -x[2])[:8]
    qgroups = [g for g, *_ in thirds_all]
    third_team = {g: t for g, t, _ in thirds_all}
    slot_assign = _assign_thirds(qgroups, [(i, s) for i, (_, _, s) in enumerate(br.third_slots)])
    slot_group = {br.third_slots[i][0]: g for i, g in slot_assign.items()}

    def slot_team(side, mid):
        if side["t"] == "W": return winners[side["g"]]
        if side["t"] == "R": return runners[side["g"]]
        if side["t"] == "3": return third_team.get(slot_group.get(mid))
        return None  # M / L resolved later

    out = []
    win_of, lose_of = {}, {}

    def play(m, home, away, rd):
        res = results.get(str(m["id"]))
        # a recorded knockout result may carry the ACTUAL teams (the projected
        # third-place slotting won't always match FIFA's real draw) — trust it.
        if res and res.get("home") and res.get("away"):
            home, away = res["home"], res["away"]
        city = m["city"]
        eh = elo.get(home, 1500) + (home_adv if match_country(city) == home else 0)
        ea = elo.get(away, 1500) + (home_adv if match_country(city) == away else 0)
        dr = eh - ea
        ph = round(_ko_winprob(dr), 3)
        model_pick = home if dr >= 0 else away          # the model's prediction
        status, result, winner = "projected", None, None
        if res:
            status, result = "final", res
            if res.get("winner"):                        # explicit (handles pens)
                winner = res["winner"]
            elif res["home_goals"] != res["away_goals"]:
                winner = home if res["home_goals"] > res["away_goals"] else away
        advancer = winner or model_pick                  # who moves on in the bracket
        win_of[m["id"]] = advancer
        lose_of[m["id"]] = away if advancer == home else home
        score = _best_score(dr if model_pick == home else -dr, elo_per_goal, total_goals, max_sup)
        out.append({
            "id": m["id"], "round": rd, "date": m.get("date"), "city": city,
            "home": home, "away": away,
            "p_home": ph, "p_away": round(1 - ph, 3),
            "pick": model_pick, "winner": winner, "score": score,
            "status": status, "result": result,
        })

    for m in r32:
        play(m, slot_team(m["home"], m["id"]), slot_team(m["away"], m["id"]), "Round of 32")
    for rd in br.rounds[1:]:
        for m in rd["matches"]:
            play(m, win_of.get(m["home"]["m"]), win_of.get(m["away"]["m"]), rd["name"])
    if br.third_place:
        tp = br.third_place
        play(tp, lose_of.get(tp["home"]["m"]), lose_of.get(tp["away"]["m"]), "Third place")

    champion = win_of.get(br.rounds[-1]["matches"][0]["id"])
    return {"matches": out, "champion": champion}
