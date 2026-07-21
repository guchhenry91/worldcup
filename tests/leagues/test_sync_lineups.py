import datetime as dt
import json

from scripts import sync_lineups


NOW = dt.datetime(2026, 8, 15, 11, 30, tzinfo=dt.timezone.utc)


class FakeClient:
    instances = []

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.calls = []
        self.instances.append(self)

    def get(self, path, **params):
        self.used += 1
        self.calls.append((path, params))
        if path == "fixtures":
            return [{
                "fixture": {"id": 999}, "league": {"id": 39},
                "teams": {"home": {"name": "Arsenal"},
                          "away": {"name": "Chelsea"}},
            }]
        return [
            {"team": {"name": "Arsenal"},
             "startXI": [{"player": {"name": f"A{i}"}} for i in range(11)],
             "substitutes": []},
            {"team": {"name": "Chelsea"},
             "startXI": [{"player": {"name": f"C{i}"}} for i in range(11)],
             "substitutes": []},
        ]


def test_lineup_fetch_is_saved_and_never_repeated(tmp_path, monkeypatch):
    news = tmp_path / "news.json"
    news.write_text(json.dumps({"PL": {}}), encoding="utf-8")
    fixture = {"league_key": "PL", "date": "2026-08-15T12:00:00Z",
               "home": "Arsenal", "away": "Chelsea"}
    monkeypatch.setenv("API_FOOTBALL_KEY", "hidden")
    monkeypatch.setattr(sync_lineups, "NEWS_PATH", news)
    monkeypatch.setattr(sync_lineups, "upcoming_fixtures", lambda now: [fixture])
    monkeypatch.setattr(sync_lineups, "Client", FakeClient)

    assert sync_lineups.main(now=NOW) == 0
    saved = json.loads(news.read_text(encoding="utf-8"))
    assert saved["PL"]["Arsenal"]["lineup_confirmed"] is True
    assert len(saved["PL"]["Chelsea"]["starters"]) == 11
    assert FakeClient.instances[-1].used == 2  # one date + one lineup request

    assert sync_lineups.main(now=NOW) == 0
    assert FakeClient.instances[-1].used == 0
