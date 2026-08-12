"""
Retrieval evaluation — does hybrid search actually beat its two halves?

Everything else in this project is an assertion. This file is the evidence.

HOW IT WORKS
Each query below is paired with a "gold" marker: a short phrase that appears
in exactly one place in the corpus. A retrieval counts as correct if the gold
marker shows up in one of the top-k returned chunks. That keeps the labels
independent of chunk IDs, so re-chunking doesn't invalidate the eval.

The queries are deliberately split into three families, because a single
average hides the whole point of hybrid search:

  paraphrase  — asks the question in words the document never uses.
                Vector search should win. BM25 has nothing to match on.
  exact       — an error code or invoice number.
                BM25 should win. Embeddings blur rare tokens together.
  mixed       — ordinary questions that share some vocabulary with the answer.

If hybrid is worth its complexity, it should track the WINNER of each family
rather than the average of the two. That is the claim this file tests.

Run:  .venv/bin/python eval.py
"""

import statistics
import sys
import time

from backend import db, ingest, search

CORPUS_DIR = "data/eval_corpus"
K = 5
METHODS = ["keyword", "vector", "rrf", "weighted"]

# (family, query, gold marker that must appear in a retrieved chunk)
QUERIES: list[tuple[str, str, str]] = [
    # ---- paraphrase: the answer never uses the query's words ----
    ("paraphrase", "how long until I get my money back", "7 working days of approval"),
    ("paraphrase", "can I stop an order that has not gone out yet", "any time before it ships"),
    ("paraphrase", "what happens if nobody is home to receive it", "attempts delivery three times"),
    ("paraphrase", "why is a light box so expensive to send", "divided by 5000"),
    ("paraphrase", "I am locked out and no longer have my phone", "recovery code or an existing trusted device"),
    ("paraphrase", "am I forced to pick a new password every few months", "We do not enforce rotation"),
    ("paraphrase", "what happens if I stop paying", "grace period of 7 days"),
    ("paraphrase", "when do you tell customers something bad happened", "within 72 hours"),
    ("paraphrase", "how fast does someone write back", "answered within one working day"),
    ("paraphrase", "switching to a cheaper plan halfway through", "start of the next billing cycle"),
    ("paraphrase", "the link says no information available", "issued at manifest, not at pickup"),
    ("paraphrase", "how hard can I hammer the endpoint", "600 requests per minute"),

    # ---- exact: rare identifiers, no semantic content ----
    ("exact", "AUTH-1203", "AUTH-1203"),
    ("exact", "SKU-99213", "SKU-99213"),
    ("exact", "PAY-4471", "PAY-4471"),
    ("exact", "ADR-8890", "ADR-8890"),
    ("exact", "BIL-5510", "BIL-5510"),
    ("exact", "BIL-6620", "BIL-6620"),
    ("exact", "SEC-3310", "SEC-3310"),
    ("exact", "SEC-4420", "SEC-4420"),
    ("exact", "SHP-1140", "SHP-1140"),
    ("exact", "SHP-2250", "SHP-2250"),
    ("exact", "INV-2291", "INV-2291"),
    ("exact", "SHP-3360", "SHP-3360"),

    # ---- mixed: ordinary questions, partial vocabulary overlap ----
    ("mixed", "warranty on accessories", "Accessories carry 6 months"),
    ("mixed", "when does a ticket escalate", "48 hours without a first response"),
    ("mixed", "bcrypt work factor", "work factor of 12"),
    ("mixed", "how long are audit logs kept", "retained for 400 days"),
    ("mixed", "credit note approval limit", "above 25000 rupees"),
    ("mixed", "customs hold clearance time", "clear within 4 working days"),
    ("mixed", "reverse pickup reimbursement", "150 rupees"),
    ("mixed", "same day delivery cities", "Bangalore and Mumbai"),
]


def build_index() -> int:
    """Wipe and rebuild from the eval corpus so runs are reproducible."""
    import os

    db.reset()
    total = 0
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not name.endswith(".txt"):
            continue
        report = ingest.ingest_file(os.path.join(CORPUS_DIR, name), name)
        total += report["chunks"]
        print(f"  indexed {name:<16} {report['chunks']:>3} chunks")
    return total


def rank_of_gold(results: list[dict], gold: str) -> int | None:
    """1-based rank of the first chunk containing the gold marker, else None."""
    needle = gold.lower()
    for i, r in enumerate(results, start=1):
        if needle in r["text"].lower():
            return i
    return None


def main() -> int:
    print("building index from", CORPUS_DIR)
    chunks = build_index()
    print(f"corpus: {chunks} chunks\n")

    # method -> family -> list of ranks (None = miss)
    ranks: dict[str, dict[str, list]] = {
        m: {f: [] for f in ("paraphrase", "exact", "mixed")} for m in METHODS
    }
    latencies: dict[str, list[float]] = {m: [] for m in METHODS}

    for family, query, gold in QUERIES:
        for method in METHODS:
            started = time.perf_counter()
            results = search.hybrid_search(query, k=K, method=method)
            latencies[method].append((time.perf_counter() - started) * 1000)
            ranks[method][family].append(rank_of_gold(results, gold))

    def recall(rs: list, at: int) -> float:
        hit = sum(1 for r in rs if r is not None and r <= at)
        return 100.0 * hit / len(rs) if rs else 0.0

    def mrr(rs: list) -> float:
        return sum(1.0 / r for r in rs if r is not None) / len(rs) if rs else 0.0

    families = ("paraphrase", "exact", "mixed")
    print(f"{'method':<10} {'paraphrase':>11} {'exact':>8} {'mixed':>8} "
          f"{'| R@1':>8} {'R@5':>7} {'MRR':>7} {'p50ms':>7} {'p95ms':>7}")
    print("-" * 82)

    summary = {}
    for method in METHODS:
        all_ranks = [r for f in families for r in ranks[method][f]]
        per_family = [recall(ranks[method][f], K) for f in families]
        lat = sorted(latencies[method])
        p50 = lat[len(lat) // 2]
        p95 = lat[int(len(lat) * 0.95)]
        summary[method] = {
            "recall@5": recall(all_ranks, K),
            "recall@1": recall(all_ranks, 1),
            "mrr": mrr(all_ranks),
            "p50_ms": p50,
            "p95_ms": p95,
            "per_family": dict(zip(families, per_family)),
        }
        print(f"{method:<10} "
              f"{per_family[0]:>10.0f}% {per_family[1]:>7.0f}% {per_family[2]:>7.0f}% "
              f"| {recall(all_ranks, 1):>6.0f}% {recall(all_ranks, K):>6.0f}% "
              f"{mrr(all_ranks):>7.3f} {p50:>7.1f} {p95:>7.1f}")

    print("-" * 82)
    best_half = max(summary["keyword"]["recall@5"], summary["vector"]["recall@5"])
    rrf_r5 = summary["rrf"]["recall@5"]
    print(f"\ncorpus            {chunks} chunks over {len(QUERIES)} labelled queries, k={K}")
    print(f"best single half  {best_half:.0f}% recall@5")
    print(f"rrf hybrid        {rrf_r5:.0f}% recall@5  "
          f"({rrf_r5 - best_half:+.0f} points over the better half)")
    print(f"keyword-only on paraphrases   {summary['keyword']['per_family']['paraphrase']:.0f}%")
    print(f"vector-only on exact codes    {summary['vector']['per_family']['exact']:.0f}%")

    import json
    with open("data/eval_results.json", "w") as f:
        json.dump({"chunks": chunks, "queries": len(QUERIES), "k": K,
                   "methods": summary}, f, indent=2)
    print("\nwrote data/eval_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
