"""Per-league configuration. One entry per competition; the engine is generic."""
from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    key: str              # our canonical key, used in paths/URLs
    name: str             # display name
    fd_code: str          # football-data.co.uk division code
    fixture_slug: str     # fixturedownload.com slug for 2026-27
    understat: str        # soccerdata/Understat league id
    fbref: str            # soccerdata/FBref league id
    n_teams: int
    relegation_spots: int
    europe_spots: int     # top-N qualifying for the Champions League
    # second-tier football-data.co.uk code, source of promoted-club priors
    fd_code2: str = ""
    # 5 completed seasons used to fit, as football-data.co.uk season codes
    history_seasons: tuple = ("2122", "2223", "2324", "2425", "2526")


LEAGUES = {
    "PL": League("PL", "Premier League", "E0", "epl-2026",
                 "ENG-Premier League", "ENG-Premier League", 20, 3, 4, "E1"),
    "LALIGA": League("LALIGA", "La Liga", "SP1", "la-liga-2026",
                     "ESP-La Liga", "ESP-La Liga", 20, 3, 4, "SP2"),
    "BUNDESLIGA": League("BUNDESLIGA", "Bundesliga", "D1", "bundesliga-2026",
                         "GER-Bundesliga", "GER-Bundesliga", 18, 2, 4, "D2"),
    "LIGUE1": League("LIGUE1", "Ligue 1", "F1", "ligue-1-2026",
                     "FRA-Ligue 1", "FRA-Ligue 1", 18, 2, 4, "F2"),
}


def get(key: str) -> League:
    if key not in LEAGUES:
        raise KeyError(f"unknown league {key!r}; known: {sorted(LEAGUES)}")
    return LEAGUES[key]
