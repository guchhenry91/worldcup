from leagues.fixtures import parse_fixtures

RAW = [
    {"MatchNumber": 1, "RoundNumber": 1, "DateUtc": "2026-08-21 19:00:00Z",
     "Location": "Emirates Stadium", "HomeTeam": "Arsenal", "AwayTeam": "Coventry",
     "HomeTeamScore": None, "AwayTeamScore": None},
    {"MatchNumber": 2, "RoundNumber": 1, "DateUtc": "2026-08-22 14:00:00Z",
     "Location": "Anfield", "HomeTeam": "Liverpool", "AwayTeam": "Man City",
     "HomeTeamScore": 2, "AwayTeamScore": 1},
]


def test_parse_fixtures_normalizes_names_and_types():
    df = parse_fixtures(RAW, "PL")
    assert len(df) == 2
    assert list(df.columns) == ["match_id", "round", "date", "venue",
                                "home", "away", "home_goals", "away_goals", "played"]
    assert df.loc[0, "home"] == "Arsenal"
    assert df.loc[1, "away"] == "Manchester City"


def test_parse_fixtures_marks_played_only_when_both_scores_present():
    df = parse_fixtures(RAW, "PL")
    assert bool(df.loc[0, "played"]) is False
    assert bool(df.loc[1, "played"]) is True
    assert df.loc[1, "home_goals"] == 2
