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


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    log.info("loading embedding model %s (first call only)", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


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
