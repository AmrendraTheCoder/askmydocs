"""
Storage layer. Two things live here:

  1. CHROMA  — holds the vectors, answers "what's closest in meaning?"
  2. chunks.json — a plain list of every chunk's text.

WHY TWO STORES?
BM25 (the keyword half of hybrid search) needs to see ALL the text to
build its index — it counts word frequencies across the whole corpus.
Chroma is built for "give me the nearest 10", not "give me everything".
So the raw text lives in a simple JSON file that BM25 can read in one go.

At real scale that JSON becomes Postgres or Elasticsearch. Same shape,
different box. Say that if they ask about scaling.
"""

import json
import logging
import os
import threading

import chromadb

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

for d in (DATA_DIR, CHROMA_DIR, UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

COLLECTION = "docs"

# A lock so two simultaneous uploads can't both read-modify-write
# chunks.json and lose each other's data. Classic race condition —
# a great thing to mention as a "dev support" war story.
_lock = threading.Lock()

_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    """Always fetch through this, never cache the object — /reset replaces it."""
    return _client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # rank by cosine similarity
    )


# ---------------------------------------------------------------
# the plain-text side
# ---------------------------------------------------------------
def load_chunks() -> list[dict]:
    if not os.path.exists(CHUNKS_FILE):
        return []
    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # File got corrupted (killed mid-write, disk full...). Don't crash
        # the whole service over it — log loudly and carry on empty.
        log.exception("chunks.json is corrupt, starting from empty")
        return []


def _save_chunks(chunks: list[dict]) -> None:
    # Write to a temp file then rename. Rename is atomic on macOS/Linux, so
    # a crash mid-write leaves the OLD good file instead of half a file.
    tmp = CHUNKS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    os.replace(tmp, CHUNKS_FILE)


# ---------------------------------------------------------------
# writing
# ---------------------------------------------------------------
def add_chunks(records: list[dict], embeddings: list[list[float]]) -> int:
    """records: [{id, text, source, page}, ...] — must line up with embeddings."""
    if not records:
        return 0

    with _lock:
        get_collection().upsert(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            embeddings=embeddings,
            metadatas=[{"source": r["source"], "page": r["page"]} for r in records],
        )

        existing = load_chunks()
        known = {c["id"] for c in existing}
        existing.extend(r for r in records if r["id"] not in known)
        _save_chunks(existing)

    log.info("stored %d chunks from %s", len(records), records[0]["source"])
    return len(records)


# ---------------------------------------------------------------
# admin / support helpers
# ---------------------------------------------------------------
def stats() -> dict:
    chunks = load_chunks()
    sources = sorted({c["source"] for c in chunks})
    return {
        "chunks": len(chunks),
        "vectors": get_collection().count(),
        "documents": len(sources),
        "sources": sources,
    }


def reset() -> None:
    """Wipe everything. Handy in development, and the kind of admin button
    that saves you an hour when support says 'the index looks wrong'."""
    with _lock:
        try:
            _client.delete_collection(COLLECTION)
        except Exception:
            log.warning("collection %s did not exist", COLLECTION)
        get_collection()
        _save_chunks([])
    log.warning("index reset — all data cleared")
