import leagues.publish as publish

_EMPTY_BEST = {"record": {"correct": 0, "wrong": 0}, "upcoming": [], "settled": [],
               "_incomplete": []}
_EMPTY_PLAYERS = {"record": {"correct": 0, "wrong": 0}, "upcoming": [], "settled": [],
                  "record_by_market": {}, "min_probability": {}, "_incomplete": []}

_STUB = lambda lg: {"league": lg, "matches": [], "table": [],
                    "missing_squads": [], "data_warnings": []}


def test_main_writes_one_atomic_file_per_league(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)
    monkeypatch.setattr(publish, "build", _STUB)
    monkeypatch.setattr(publish, "build_best_picks", lambda: _EMPTY_BEST)
    monkeypatch.setattr(publish, "build_player_picks", lambda: _EMPTY_PLAYERS)
    publish.main([])                       # no arg -> all leagues
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    # best.json is the cross-league high-confidence board, written after the leagues
    assert written == ["best.json", "bundesliga.json", "laliga.json",
                       "ligue1.json", "pl.json", "player_picks.json",
                       "record_history.json"]
    assert not list(tmp_path.glob("*.tmp"))          # no leftover temp files


def test_one_league_failing_does_not_block_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)

    def flaky(lg):
        if lg == "LALIGA":
            raise RuntimeError("simulated fetch failure")
        return _STUB(lg)

    monkeypatch.setattr(publish, "build", flaky)
    monkeypatch.setattr(publish, "build_best_picks", lambda: _EMPTY_BEST)
    monkeypatch.setattr(publish, "build_player_picks", lambda: _EMPTY_PLAYERS)
    # Seed a PRE-EXISTING laliga.json. Asserting against an empty directory could
    # not tell "correctly skipped" from "silently left stale", which is the actual
    # hazard -- an aborted league leaving last week's file for the gate to pass.
    (tmp_path / "laliga.json").write_text('{"league": "STALE"}', encoding="utf-8")
    import pytest
    with pytest.raises(RuntimeError, match="3/4"):
        publish.main([])                   # partial files stay local; no deployment
    written = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert "pl" in written and "bundesliga" in written and "ligue1" in written
    # the failing league's file is untouched, NOT overwritten with partial data
    import json as _json
    assert _json.loads((tmp_path / "laliga.json").read_text())["league"] == "STALE"
    assert not (tmp_path / "best.json").exists()
    assert not (tmp_path / "player_picks.json").exists()


def test_single_league_arg_writes_only_that_file(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)
    monkeypatch.setattr(publish, "build", _STUB)
    monkeypatch.setattr(publish, "build_best_picks", lambda: _EMPTY_BEST)
    monkeypatch.setattr(publish, "build_player_picks", lambda: _EMPTY_PLAYERS)
    publish.main(["PL"])                    # quick-iteration path
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["pl.json"]
