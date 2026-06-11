#!/usr/bin/env python3
"""Verify /recall respects max_tokens across the budget range.

For each budget, issues a fact-heavy recall and reports
approx_tokens(context) / budget. The contract allows 2x; this asserts <= 2.0
and prints the measured ratio (committed runs are quoted in CHANGELOG /
README). Run against a populated service:

    .venv/bin/python scripts/budget_sweep.py [--base-url http://127.0.0.1:8080]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory_service.tokens import approx_tokens  # noqa: E402

BUDGETS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--user", default="fx-maya")
    ap.add_argument("--query", default="Where does the user work and what pets do they have?")
    args = ap.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=30)
    print(f"{'budget':>7} {'used':>6} {'ratio':>6}  context-head")
    worst = 0.0
    for budget in BUDGETS:
        r = client.post("/recall", json={
            "query": args.query, "session_id": "budget-sweep",
            "user_id": args.user, "max_tokens": budget,
        })
        r.raise_for_status()
        ctx = r.json()["context"]
        used = approx_tokens(ctx)
        ratio = used / budget
        worst = max(worst, ratio)
        print(f"{budget:7} {used:6} {ratio:6.2f}  {ctx[:60].replace(chr(10), ' / ')}")
        assert ratio <= 2.0, f"blew the 2x contract bound at max_tokens={budget}"
    print(f"\nworst ratio: {worst:.2f} (contract bound: 2.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
