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

from memory_service import config, db, embeddings, retrieval, store  # noqa: E402


def _term_support(p: dict, diag: dict) -> tuple[float, str]:
    """Mirror the gate's Rule B computation for this probe: over the top-5
    ambiguous-zone items, the best (content-term ⋅ item) cosine, and which
    term supplied it. Returns (0.0, "-") when the gate never reaches Rule B
    for this probe (an item cleared DENSE_FLOOR, or the zone is empty)."""
    zone: list[tuple[str, float]] = []
    for sims in (diag["dense_memory"], diag["dense_turn"]):
        for item_id, sim in sims.items():
            if sim >= config.DENSE_FLOOR:
                return 0.0, "-"  # Rule A decided; Rule B never ran
            if sim >= config.DENSE_FLOOR_LOW:
                zone.append((item_id, sim))
    if not zone:
        return 0.0, "-"
    terms = retrieval._content_terms(p["query"])
    best_support, best_term = 0.0, "-"
    for item_id, _sim in sorted(zone, key=lambda x: -x[1])[:5]:
        item_vec = retrieval._item_embedding(item_id)
        if item_vec is None:
            continue
        for t in terms:
            tv = embeddings.embed_query(t)
            if tv is None:
                continue
            s = float(tv @ item_vec)
            if s > best_support:
                best_support, best_term = s, t
    return best_support, best_term


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "memory.db"))
    ap.add_argument("--fixture", default=str(ROOT / "fixtures" / "conversations.json"))
    args = ap.parse_args()

    db.init(args.db)
    fixture = json.loads(Path(args.fixture).read_text())

    print(f"{'probe':6} {'category':14} {'max_mem':>8} {'max_turn':>8} {'support':>8} {'via':>10} {'gate':>6}")
    noise_best: list[float] = []
    signal_best: list[float] = []
    noise_support: list[float] = []
    signal_support: list[float] = []
    for p in fixture["probes"]:
        owner = store.resolve_owner(p["user_id"], p["session_id"])
        r = retrieval.retrieve(owner, p["query"])
        d = r["diagnostics"]
        best = max(d["max_dense_memory"], d["max_dense_turn"])
        support, via = _term_support(p, d)
        print(f"{p['id']:6} {p['category']:14} {d['max_dense_memory']:8.3f} "
              f"{d['max_dense_turn']:8.3f} {support:8.3f} {via:>10} "
              f"{'PASS' if r['relevant'] else 'gated':>6}")
        if p.get("expect_empty"):
            noise_best.append(best)
            if via != "-":
                noise_support.append(support)
        else:
            signal_best.append(best)
            if via != "-":
                signal_support.append(support)

    print(f"\nnoise/cold probes, best sims (gate must REJECT): {sorted(round(x, 3) for x in noise_best)}")
    print(f"signal probes, lowest 5 (gate must PASS):         {sorted(round(x, 3) for x in signal_best)[:5]}")
    print(f"\nRule-B term support — noise (must stay < TERM_FLOOR={config.TERM_FLOOR}): "
          f"{sorted(round(x, 3) for x in noise_support)}")
    print(f"Rule-B term support — signal (must reach TERM_FLOOR where Rule B decides): "
          f"{sorted(round(x, 3) for x in signal_support)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
