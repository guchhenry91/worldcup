"""Match day: pull results, re-grade the FROZEN picks, redeploy.

Same shape as the weekly job -- publish.build() already grades every played
fixture against the pick that was locked before kickoff -- but it runs on match
evenings so the record updates the same night rather than the following week.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    try:
        from leagues import publish
        publish.main()
    except Exception as exc:
        print(f"ABORT: league publish failed ({exc}); nothing deployed", file=sys.stderr)
        return 1

    # Same gate as the weekly job: the numbers must make sense before they ship.
    from scripts import sanity_check
    if sanity_check.main() != 0:
        print("ABORT: published payload failed sanity checks; nothing deployed",
              file=sys.stderr)
        return 1

    return subprocess.call(
        [sys.executable, str(ROOT / "deploy.py"), "auto update: leagues match-day results",
         "--league-data"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
