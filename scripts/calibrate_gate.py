#!/usr/bin/env python3
"""Calibrate the relevance-gate floors against the fixture probes.

For every probe, computes the max query->memory and query->turn dense
similarity in that probe's owner scope, plus whether any single item clears
the per-item evidence rule. Prints the distributions the floors must
separate: noise/cold probes (should fail the gate) vs everything else
(should pass).

Run against a populated local store (after `scripts/selfeval.py` has
ingested, or pointing MEMORY_DB_PATH at the docker volume's file):

    PYTHONPATH=src .venv/bin/python scripts/calibrate_gate.py [--db data/memory.db]

If you change MEMORY_EMBEDDING_MODEL, re-run this and pick new
MEMORY_DENSE_FLOOR / MEMORY_DENSE_FLOOR_LOW values that separate the two
distributions (see CHANGELOG v0.4/v0.7 for the readings behind the
committed defaults).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory_service import db, embeddings, retrieval, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "memory.db"))
    ap.add_argument("--fixture", default=str(ROOT / "fixtures" / "conversations.json"))
    args = ap.parse_args()

    db.init(args.db)
    fixture = json.loads(Path(args.fixture).read_text())

    print(f"{'probe':6} {'category':14} {'max_mem':>8} {'max_turn':>8} {'gate':>6}")
    noise_best: list[float] = []
    signal_best: list[float] = []
    for p in fixture["probes"]:
        owner = store.resolve_owner(p["user_id"], p["session_id"])
        r = retrieval.retrieve(owner, p["query"])
        d = r["diagnostics"]
        best = max(d["max_dense_memory"], d["max_dense_turn"])
        print(f"{p['id']:6} {p['category']:14} {d['max_dense_memory']:8.3f} "
              f"{d['max_dense_turn']:8.3f} {'PASS' if r['relevant'] else 'gated':>6}")
        if p.get("expect_empty"):
            noise_best.append(best)
        else:
            signal_best.append(best)

    print(f"\nnoise/cold probes, best sims (gate must REJECT): {sorted(round(x, 3) for x in noise_best)}")
    print(f"signal probes, lowest 5 (gate must PASS):         {sorted(round(x, 3) for x in signal_best)[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
