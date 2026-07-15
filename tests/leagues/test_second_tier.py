import io

from leagues.second_tier import table_strengths, promoted_deviations, ATTACK_MAP

# A tiny second-tier season: Coventry dominates, Mid is average, Weak sides poor.
CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
    + "".join(f"E1,01/01/2026,Coventry,Weak{i},3,0\n" for i in range(5))
    + "".join(f"E1,01/01/2026,Mid,Weak{i},1,1\n" for i in range(5))
    + "".join(f"E1,01/01/2026,Weak{i},Mid,0,2\n" for i in range(5))
)


def test_table_strengths_rank_by_goal_form():
    s = table_strengths(io.StringIO(CSV), "PL")
    assert s["Coventry"]["gf_pg"] > s["Mid"]["gf_pg"]      # scores more
    assert s["Coventry"]["ga_pg"] < s["Mid"]["ga_pg"]      # concedes less
    assert s["Coventry"]["lg"] > 0                          # league baseline present


def test_promoted_attack_prior_is_below_average_even_for_the_champion():
    """The fitted map has a strong negative intercept: even a dominant second-tier
    attack lands BELOW the top-flight scoring average -- promoted clubs are weak."""
    s = table_strengths(io.StringIO(CSV), "PL")
    dev = promoted_deviations(s, ["Coventry"])
    att, dfn = dev["Coventry"]
    assert att < 0                     # below top-flight average, despite dominating below
    assert att > ATTACK_MAP[1]         # ...but its strong form lifts it above the intercept


def test_stronger_second_tier_attack_gives_a_higher_prior():
    s = table_strengths(io.StringIO(CSV), "PL")
    dev = promoted_deviations(s, ["Coventry", "Mid"])
    assert dev["Coventry"][0] > dev["Mid"][0]   # attack signal survives (positive slope)


def test_defence_prior_is_a_constant_not_a_function_of_second_tier_defence():
    """2nd-tier defence has no predictive signal (r=-0.04): the prior is a constant,
    so a great and a leaky second-tier defence get the SAME top-flight defence prior."""
    s = table_strengths(io.StringIO(CSV), "PL")
    dev = promoted_deviations(s, ["Coventry", "Mid"])
    assert dev["Coventry"][1] == dev["Mid"][1]  # identical defence prior
    assert dev["Coventry"][1] > 0               # concedes above average (worse)


def test_unknown_team_is_simply_absent_not_an_error():
    s = table_strengths(io.StringIO(CSV), "PL")
    out = promoted_deviations(s, ["Coventry", "NeverPlayed"])
    assert "Coventry" in out and "NeverPlayed" not in out
