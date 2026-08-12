"""
The embedding model, loaded once and reused.

WHY A SINGLETON:
Loading the model takes ~2 seconds and ~100MB of RAM. If you loaded it
inside every request, every search would take 2 extra seconds and your
memory would balloon. So you load it once, on first use, and keep it.

@lru_cache(maxsize=1) is the cheap way to say "run this function once,
then hand back the same result forever".
"""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# 384 dimensions, ~90MB, good enough for almost everything.
# Upgrade path if quality is poor: "BAAI/bge-small-en-v1.5" (same size,
# usually better) or "all-mpnet-base-v2" (768 dims, slower, stronger).
MODEL_NAME = "all-MiniLM-L6-v2"

# CPU, deliberately — not an oversight.
#
# sentence-transformers picks the "best" device it can find, which on a Mac
# means Metal (mps). For THIS workload that is the wrong default. Measured
# over 60 single-query encodes, after burn-in:
#
#     mps   p50  6.89ms   p95  18.75ms   max 151.33ms
#     cpu   p50  3.95ms   p95   5.23ms   max  15.52ms
#
# The model is tiny (22M params) and a search request embeds ONE short
# string, so the work is far too small to amortise the round trip to the
# GPU — the transfer and MPS scheduling cost more than the matmul saves.
# CPU is faster at the median and, more importantly, ~10x tighter in the
# tail, which is what a latency alert actually keys off.
#
# This flips for BATCH work: ingesting a large PDF embeds hundreds of
# chunks at once, and there mps wins. If bulk ingest ever becomes the
# bottleneck, make the device per-call rather than per-process.
DEVICE = "cpu"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load once, and prefer the local cache over the network.

    local_files_only avoids a hub round-trip at boot on a machine that
    already has the weights, and means startup doesn't depend on
    huggingface.co being reachable. It is NOT what fixes slow first
    searches — see warmup() below for that.
    """
    log.info("loading embedding model %s (first call only)", MODEL_NAME)
    try:
        return SentenceTransformer(MODEL_NAME, device=DEVICE, local_files_only=True)
    except Exception:
        log.info("%s not in local cache — downloading once", MODEL_NAME)
        return SentenceTransformer(MODEL_NAME, device=DEVICE)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of strings into a list of number-lists.

    Always batch. Calling this once with 100 texts is far faster than
    calling it 100 times with 1 text — the model processes them together.
    """
    if not texts:
        return []
    return get_model().encode(texts, show_progress_bar=False).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def warmup(rounds: int = 3) -> None:
    """Force the lazy work that otherwise lands on the first real user.

    Loading the model is not the same as running it. Measured on this
    machine, with the model fully loaded, the first four encode() calls
    took 4293ms, 1289ms, 906ms and 429ms before settling at a steady
    ~7ms. That is PyTorch initialising kernels and its allocator on first
    forward pass, and it decays over a few calls rather than one.

    Loading alone at startup therefore fixes almost nothing — the cliff
    just moves to whoever searches first.

    This warms only the embedder. The search path has its own cold costs
    (BM25 index build, Chroma's first query), so the service warms the
    whole path at startup — see search.warmup().
    """
    get_model()
    for _ in range(rounds):
        embed_one("warmup")
