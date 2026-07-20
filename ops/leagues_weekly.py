"""Weekly: refresh data, re-fit, re-sim, republish, deploy.

Abort-on-failure: a failed fetch must NEVER ship a stale-but-fresh-looking file.
That is exactly how the WC app once published picks that had to be voided.
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

    # Gate the deploy on the published NUMBERS making sense, not just the code
    # running. Catches the class of bug unit tests miss: a pick contradicting its
    # own scoreline, an eliminated team still favoured, a table that doesn't add up.
    from scripts import sanity_check
    if sanity_check.main() != 0:
        print("ABORT: published payload failed sanity checks; nothing deployed",
              file=sys.stderr)
        return 1

    return subprocess.call(
        [sys.executable, str(ROOT / "deploy.py"), "auto update: leagues weekly refresh",
         "--league-data"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
