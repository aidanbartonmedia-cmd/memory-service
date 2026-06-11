#!/usr/bin/env python3
"""Self-eval harness: ingest fixture conversations, run recall probes, report quality.

This is the iteration loop. Run after every change:

    .venv/bin/python scripts/selfeval.py [--base-url http://127.0.0.1:8080]

Scoring rules per probe:
  expect_empty             -> context must be exactly empty ("" and no citations)
  expect_any               -> at least one substring present in context (case-insensitive)
  expect_all               -> all substrings present
  expect_history_any       -> substring present in context (as history) AND the user's
                              /memories must contain an INACTIVE memory mentioning it
                              (supersession chain check)
  expect_absent_as_current -> the user's /memories must NOT contain an ACTIVE memory
                              whose value contains this string

Writes a JSON result file under selfeval-results/ for CHANGELOG citation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def ingest(client: httpx.Client, fixture: dict) -> float:
    """Delete fixture users, then ingest all conversations. Returns total ingest seconds."""
    t0 = time.monotonic()
    for convo in fixture["conversations"]:
        r = client.delete(f"/users/{convo['user_id']}")
        assert r.status_code in (204, 404), f"cleanup failed: {r.status_code} {r.text}"
    for convo in fixture["conversations"]:
        for session in convo["sessions"]:
            for turn in session["turns"]:
                r = client.post(
                    "/turns",
                    json={
                        "session_id": session["session_id"],
                        "user_id": convo["user_id"],
                        "messages": turn["messages"],
                        "timestamp": turn["timestamp"],
                        "metadata": {},
                    },
                )
                assert r.status_code == 201, f"/turns failed: {r.status_code} {r.text}"
    return time.monotonic() - t0


def run_probe(client: httpx.Client, probe: dict) -> dict:
    t0 = time.monotonic()
    r = client.post(
        "/recall",
        json={
            "query": probe["query"],
            "session_id": probe["session_id"],
            "user_id": probe["user_id"],
            "max_tokens": probe.get("max_tokens", 512),
        },
    )
    latency_ms = (time.monotonic() - t0) * 1000
    result = {
        "id": probe["id"],
        "category": probe["category"],
        "query": probe["query"],
        "latency_ms": round(latency_ms, 1),
        "checks": [],
        "passed": True,
    }
    if r.status_code != 200:
        result["checks"].append({"check": "http_200", "ok": False, "detail": r.status_code})
        result["passed"] = False
        return result

    body = r.json()
    context = body.get("context", "")
    citations = body.get("citations", [])
    result["context"] = context

    def check(name: str, ok: bool, detail: str = "") -> None:
        result["checks"].append({"check": name, "ok": ok, "detail": detail})
        if not ok:
            result["passed"] = False

    if probe.get("expect_empty"):
        check("empty_context", context.strip() == "" and citations == [],
              f"context={context[:120]!r} citations={len(citations)}")
        return result

    for needle_group in [probe.get("expect_any")] if probe.get("expect_any") else []:
        ok = any(contains(context, n) for n in needle_group)
        check("expect_any", ok, f"none of {needle_group} in context" if not ok else "")
    for needle in probe.get("expect_all", []):
        check("expect_all", contains(context, needle), f"{needle!r} missing" if not contains(context, needle) else "")

    # History + supersession-chain checks need the memories endpoint
    if probe.get("expect_history_any") or probe.get("expect_absent_as_current"):
        mr = client.get(f"/users/{probe['user_id']}/memories")
        memories = mr.json().get("memories", []) if mr.status_code == 200 else []
        for needle_group in [probe.get("expect_history_any")] if probe.get("expect_history_any") else []:
            in_ctx = any(contains(context, n) for n in needle_group)
            check("history_in_context", in_ctx, f"none of {needle_group} in context" if not in_ctx else "")
            inactive_hit = any(
                not m.get("active", True) and any(contains(str(m.get("value", "")), n) for n in needle_group)
                for m in memories
            )
            check("superseded_memory_exists", inactive_hit,
                  f"no inactive memory mentioning {needle_group}" if not inactive_hit else "")
        for needle in probe.get("expect_absent_as_current", []):
            active_hit = any(
                m.get("active", True) and contains(str(m.get("value", "")), needle) for m in memories
            )
            check("not_active_memory", not active_hit, f"ACTIVE memory still asserts {needle!r}" if active_hit else "")

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--fixture", default=str(ROOT / "fixtures" / "conversations.json"))
    ap.add_argument("--skip-ingest", action="store_true", help="probe only (data already ingested)")
    ap.add_argument("--label", default="", help="label stored in the result file (e.g. v0.3)")
    args = ap.parse_args()

    fixture = json.loads(Path(args.fixture).read_text())
    headers = {}
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=120.0)

    r = client.get("/health")
    assert r.status_code == 200, f"service not healthy at {args.base_url}"

    ingest_seconds = None
    if not args.skip_ingest:
        print("Ingesting fixture conversations...")
        ingest_seconds = ingest(client, fixture)
        print(f"Ingest complete in {ingest_seconds:.1f}s")

    results = [run_probe(client, p) for p in fixture["probes"]]

    by_cat: dict[str, list[dict]] = {}
    for res in results:
        by_cat.setdefault(res["category"], []).append(res)

    print(f"\n{'probe':6} {'category':14} {'pass':5} {'ms':>7}  detail")
    print("-" * 80)
    for res in results:
        fails = "; ".join(str(c["detail"]) for c in res["checks"] if not c["ok"])
        print(f"{res['id']:6} {res['category']:14} {'PASS' if res['passed'] else 'FAIL':5} "
              f"{res['latency_ms']:7.0f}  {fails[:90]}")

    print("\nPer category:")
    for cat, items in sorted(by_cat.items()):
        n_pass = sum(1 for i in items if i["passed"])
        print(f"  {cat:14} {n_pass}/{len(items)}")

    total_pass = sum(1 for r_ in results if r_["passed"])
    score = total_pass / len(results)
    lat = sorted(r_["latency_ms"] for r_ in results)
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    print(f"\nOVERALL: {total_pass}/{len(results)} = {score:.2f}   recall p50={p50:.0f}ms p95={p95:.0f}ms")

    outdir = ROOT / "selfeval-results"
    outdir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outfile = outdir / f"{stamp}{'-' + args.label if args.label else ''}.json"
    outfile.write_text(json.dumps({
        "label": args.label,
        "timestamp": stamp,
        "score": score,
        "passed": total_pass,
        "total": len(results),
        "recall_latency_ms": {"p50": p50, "p95": p95},
        "ingest_seconds": ingest_seconds,
        "by_category": {cat: f"{sum(1 for i in items if i['passed'])}/{len(items)}" for cat, items in by_cat.items()},
        "results": results,
    }, indent=2))
    print(f"Wrote {outfile}")
    return 0 if score == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
