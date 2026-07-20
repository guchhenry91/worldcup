import json
from pathlib import Path

from scripts.roster_integrity_check import audit


ROOT = Path(__file__).resolve().parents[2]


def test_roster_snapshot_covers_every_configured_club_without_duplicates():
    payload = json.loads(
        (ROOT / "data-raw" / "leagues" / "rosters.json").read_text(encoding="utf-8"))
    errors, _warnings = audit(payload)
    assert errors == []


def test_roster_snapshot_is_dated_and_documents_that_it_is_provisional():
    payload = json.loads(
        (ROOT / "data-raw" / "leagues" / "rosters.json").read_text(encoding="utf-8"))
    assert payload["_verified_at"]
    assert payload["_source"]
    assert payload["_provisional"]
