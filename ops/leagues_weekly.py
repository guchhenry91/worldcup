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

    return subprocess.call(
        [sys.executable, str(ROOT / "deploy.py"), "auto update: leagues weekly refresh"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
