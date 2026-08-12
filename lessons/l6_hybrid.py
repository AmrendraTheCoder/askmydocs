"""
LESSON 6 — HYBRID SEARCH. This is the lesson that gets you the job.
Run me:  .venv/bin/python lessons/l6_hybrid.py

The job ad said "Hybrid & Vector DB". THIS FILE IS THAT LINE OF THE JOB AD.

THE PROBLEM:
  BM25 gives scores like    0.0, 3.7, 12.4   (unbounded, weird range)
  Vectors give scores like  0.31, 0.88, 0.52 (always -1 to 1)
You cannot just add those together — BM25 would bully the vector score.

So you need to FUSE them. There are exactly two answers people use.
Know both. Most candidates know zero.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

docs = [
    "Refunds are processed within 7 working days.",
    "You can cancel your order before it ships.",
    "Our office is open Monday to Friday, 9am to 6pm.",
    "Delivery usually takes 3 to 5 days in metro cities.",
    "Error code SKU-99213 means the item is out of stock.",
    "Money is returned to the original payment method automatically.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_vectors = model.encode(docs)
bm25 = BM25Okapi([d.lower().replace(".", "").split() for d in docs])


def keyword_scores(query):
    return np.array(bm25.get_scores(query.lower().split()))


def vector_scores(query):
    qv = model.encode(query)
    # cosine similarity against every document at once
    sims = doc_vectors @ qv / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(qv))
    return sims


# ===============================================================
# METHOD 1 — WEIGHTED SCORE FUSION  (normalise, then blend)
# ===============================================================
def normalize(scores):
    """Squash any score range into 0..1 so the two are comparable.
    (value - min) / (max - min)  — called min-max normalisation."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)   # all equal -> all zero, avoids divide-by-zero
    return (scores - lo) / (hi - lo)


def hybrid_weighted(query, alpha=0.5):
    """alpha = how much you trust the VECTOR side.
       alpha=1.0 -> pure semantic.  alpha=0.0 -> pure keyword."""
    k = normalize(keyword_scores(query))
    v = normalize(vector_scores(query))
    final = alpha * v + (1 - alpha) * k
    return sorted(zip(final, k, v, docs), reverse=True)


# ===============================================================
# METHOD 2 — RECIPROCAL RANK FUSION (RRF)
# Ignore the scores completely. Only use the POSITION in each list.
#   score = 1 / (k + rank)      with k=60 by convention
# Rank 1 -> 1/61, rank 2 -> 1/62 ... add them across both lists.
#
# WHY IT'S BETTER: no normalisation, no tuning, immune to one engine
# producing insane score ranges. This is what Elasticsearch and Qdrant
# ship as their default fusion. Saying "I used RRF" reads as senior.
# ===============================================================
def hybrid_rrf(query, k=60):
    fused = np.zeros(len(docs))
    for scores in (keyword_scores(query), vector_scores(query)):
        order = np.argsort(-scores)          # indices, best first
        for rank, doc_idx in enumerate(order, start=1):
            fused[doc_idx] += 1 / (k + rank)
    return sorted(zip(fused, docs), reverse=True)


# ===============================================================
# SEE IT WORK
# ===============================================================
def show(query):
    print("\n" + "=" * 66)
    print("QUERY:", query)
    print("=" * 66)
    print(f"{'final':>7} {'kw':>6} {'vec':>6}   document")
    for final, k, v, doc in hybrid_weighted(query)[:4]:
        print(f"{final:7.3f} {k:6.2f} {v:6.2f}   {doc}")
    print("  RRF ranking:")
    for score, doc in hybrid_rrf(query)[:3]:
        print(f"    {score:.4f}  {doc}")


# Case A: pure meaning. Keyword side is blind, vector side carries it.
show("when do I get my money back")

# Case B: exact code. Vector side is fuzzy, keyword side carries it.
show("SKU-99213")

# Case C: mixed — a real user query. Both contribute.
show("cancel order delivery time")


print("""
==================================================================
WHAT TO SAY IN THE INTERVIEW (say it exactly like this)
==================================================================
"I ran both retrievers in parallel — BM25 over the raw text and a
 dense vector search over MiniLM embeddings — then fused them. I
 started with min-max normalisation and a weighted blend, alpha 0.5,
 but switched to Reciprocal Rank Fusion because it doesn't need
 normalisation or tuning and it's robust when one retriever returns a
 weird score range."

Then, if they push: "Pure vector search misses exact tokens like SKU
 codes or names; pure keyword misses paraphrases. Hybrid covers both.
 Next step would be a cross-encoder reranker over the top 50 results."

THAT LAST WORD — RERANKER — is the natural follow-up question.
Answer: a cross-encoder reads the query and ONE document together and
scores relevance properly. Slow, so you only run it on the top ~50 that
hybrid already shortlisted. Model: cross-encoder/ms-marco-MiniLM-L-6-v2.
==================================================================
""")
