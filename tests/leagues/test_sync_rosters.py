import io
import json

from scripts import sync_rosters


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_roster_request_retries_transient_failure(monkeypatch):
    calls = []

    def open_url(_request, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise OSError("temporary TLS failure")
        return _Response(b'{"ok": true}')

    monkeypatch.setattr(sync_rosters.urllib.request, "urlopen", open_url)
    sleeps = []
    assert sync_rosters.get_json("https://example.test", attempts=3,
                                 sleeper=sleeps.append) == {"ok": True}
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_failed_refresh_retains_complete_verified_snapshot(tmp_path, monkeypatch):
    out = tmp_path / "rosters.json"
    old = {"_verified_at": "2026-07-21T10:00:00+00:00"}
    old.update({key: {"Club": {"players": []}} for key in sync_rosters.LEAGUES})
    out.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(sync_rosters, "OUT", out)
    monkeypatch.setattr(sync_rosters, "fetch_league",
                        lambda *_args: (_ for _ in ()).throw(OSError("TLS")))

    assert sync_rosters.main() == 0
    assert json.loads(out.read_text(encoding="utf-8")) == old
