import leagues.publish as publish

_STUB = lambda lg: {"league": lg, "matches": [], "table": [],
                    "missing_squads": [], "data_warnings": []}


def test_main_writes_one_atomic_file_per_league(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)
    monkeypatch.setattr(publish, "build", _STUB)
    publish.main([])                       # no arg -> all leagues
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == ["bundesliga.json", "laliga.json", "ligue1.json", "pl.json"]
    assert not list(tmp_path.glob("*.tmp"))          # no leftover temp files


def test_one_league_failing_does_not_block_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)

    def flaky(lg):
        if lg == "LALIGA":
            raise RuntimeError("simulated fetch failure")
        return _STUB(lg)

    monkeypatch.setattr(publish, "build", flaky)
    publish.main([])                       # must not raise
    written = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert "laliga" not in written         # the failing one is skipped
    assert "pl" in written and "bundesliga" in written and "ligue1" in written


def test_single_league_arg_writes_only_that_file(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)
    monkeypatch.setattr(publish, "build", _STUB)
    publish.main(["PL"])                    # quick-iteration path
    assert [p.name for p in tmp_path.glob("*.json")] == ["pl.json"]
