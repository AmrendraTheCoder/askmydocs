"""
HYBRID SEARCH — the heart of the project, and the line on the job ad.

Two retrievers run over the same chunks:

  KEYWORD (BM25)  — good at exact tokens: names, codes, IDs, rare words
  VECTOR (Chroma) — good at meaning: paraphrases, fuzzy questions

Then they get FUSED into one ranking. Two fusion methods are implemented:

  "rrf"      Reciprocal Rank Fusion. Uses only each result's POSITION,
             not its score. No normalisation, no tuning, robust. Default.
  "weighted" Min-max normalise both score lists into 0-1, then blend with
             alpha. Tunable, more explainable, but sensitive to outliers.
"""

import logging
import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi

from . import db, embedder

log = logging.getLogger(__name__)

RRF_K = 60          # the standard constant. Dampens how much rank 1 dominates.

_bm25_cache: dict = {"signature": None, "index": None, "ids": None}
_cache_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    """Lowercase and keep only letters/numbers/hyphens.

    Keeping hyphens matters: it preserves 'INV-2291' and 'SKU-99213' as
    ONE token. Split those and your exact-code search quietly dies.
    """
    return re.findall(r"[a-z0-9\-]+", text.lower())


def _get_bm25(chunks: list[dict]):
    """Rebuild the BM25 index only when the corpus actually changed.

    BM25Okapi has no incremental 'add one document' — it recomputes word
    statistics over everything. So we rebuild on change and cache otherwise.
    Fine up to ~100k chunks; past that you move to Elasticsearch/OpenSearch.
    """
    signature = (len(chunks), chunks[-1]["id"] if chunks else None)
    with _cache_lock:
        if _bm25_cache["signature"] != signature:
            log.info("rebuilding BM25 index over %d chunks", len(chunks))
            _bm25_cache["index"] = BM25Okapi([tokenize(c["text"]) for c in chunks])
            _bm25_cache["ids"] = [c["id"] for c in chunks]
            _bm25_cache["signature"] = signature
        return _bm25_cache["index"]


def _normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)     # all identical -> all zero, no div by 0
    return (scores - lo) / (hi - lo)


def _rrf(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores into 1/(K + rank) contributions."""
    out = np.zeros_like(scores, dtype=float)
    for rank, idx in enumerate(np.argsort(-scores), start=1):
        out[idx] = 1.0 / (RRF_K + rank)
    return out


def hybrid_search(query: str, k: int = 5, alpha: float = 0.5,
                  method: str = "rrf") -> list[dict]:
    """alpha only applies to method='weighted'. 1.0 = pure vector, 0.0 = pure keyword."""
    chunks = db.load_chunks()
    if not chunks:
        return []

    ids = [c["id"] for c in chunks]

    # ---- keyword side: BM25 scores every chunk ----
    keyword = np.array(_get_bm25(chunks).get_scores(tokenize(query)), dtype=float)

    # ---- vector side: Chroma returns only its top N ----
    # We ask for more than k so the fusion has room to reorder. Anything
    # Chroma didn't return gets 0 — it wasn't close enough to matter.
    depth = min(len(chunks), max(k * 5, 25))
    res = db.get_collection().query(
        query_embeddings=[embedder.embed_one(query)],
        n_results=depth,
    )
    # Chroma returns DISTANCE. With cosine space: similarity = 1 - distance.
    sims = {doc_id: 1.0 - dist
            for doc_id, dist in zip(res["ids"][0], res["distances"][0])}
    vector = np.array([sims.get(i, 0.0) for i in ids], dtype=float)

    # ---- fuse ----
    if method == "weighted":
        kw_n, vec_n = _normalize(keyword), _normalize(vector)
        final = alpha * vec_n + (1 - alpha) * kw_n
    elif method == "rrf":
        final = _rrf(keyword) + _rrf(vector)
    elif method == "keyword":
        final = keyword
    elif method == "vector":
        final = vector
    else:
        raise ValueError(f"unknown method '{method}' "
                         "(use rrf, weighted, keyword or vector)")

    top = np.argsort(-final)[:k]
    results = []
    for rank, i in enumerate(top, start=1):
        if final[i] <= 0:
            continue                      # nothing matched at all — don't pad
        chunk = chunks[i]
        results.append({
            "rank": rank,
            "score": round(float(final[i]), 5),
            "keyword_score": round(float(keyword[i]), 3),
            "vector_score": round(float(vector[i]), 3),
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
        })

    log.info("query=%r method=%s hits=%d", query, method, len(results))
    return results


def build_answer(query: str, results: list[dict]) -> str:
    """The 'G' in RAG, minus the LLM.

    A real RAG app posts these chunks plus the question to an LLM. This
    project stops one step short on purpose: no API key, no cost, nothing
    invented. Every word shown to the user is text that really exists in
    their documents.

    To make it a full RAG app, send this exact prompt to any LLM.
    """
    if not results:
        return "I couldn't find anything relevant in the indexed documents."

    context = "\n\n".join(
        f"[{r['source']} p{r['page']}] {r['text']}" for r in results
    )
    return (
        f"Question: {query}\n\n"
        f"Top {len(results)} passages from your documents:\n\n{context}"
    )
