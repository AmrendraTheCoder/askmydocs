# AskMyDocs

Upload documents or images, ask questions in plain English, get the exact passage back — using **hybrid retrieval** (BM25 keyword search fused with dense vector search) over a local vector database.

Built as a learning project for a Python backend role. Every part of it is small enough to read in one sitting.

---

## What it does

- **Ingest** PDFs, text files, or images. Images go through an OpenCV cleanup pipeline and Tesseract OCR.
- **Chunk** documents by paragraph (not blind word windows) with overlap across boundaries.
- **Embed** each chunk with `all-MiniLM-L6-v2`, store the vectors in Chroma.
- **Search** with four selectable strategies — `rrf`, `weighted`, `vector`, `keyword` — so you can see exactly what each half of the hybrid contributes.
- **Search by photo** — upload a picture, its text is OCR'd and used as the query.

## Why hybrid

The two retrievers fail on opposite things:

| Query | BM25 | Vector | Hybrid |
|---|---|---|---|
| `"when do I get my money back"` | blind — no shared words | finds it | ✅ |
| `"AUTH-1203"` | exact hit | fuzzy, returns near-misses | ✅ |

Fusion is **Reciprocal Rank Fusion** by default (`1/(k + rank)`, k=60) — no score normalisation and no tuning needed. A normalised weighted blend is also implemented and selectable.

## Architecture

```
                    ┌──────────────┐
                    │ Streamlit UI │
                    └──────┬───────┘
                           │ REST
                    ┌──────▼───────┐
                    │   FastAPI    │
                    └──────┬───────┘
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼─────┐   ┌──────▼──────┐  ┌──────▼──────┐
   │  ingest.py │   │  vision.py  │  │  search.py  │
   │ chunk+embed│   │ OpenCV+OCR  │  │   hybrid    │
   └──────┬─────┘   └─────────────┘  └──────┬──────┘
          │                                 │
   ┌──────▼──────────────┐         ┌────────▼────────┐
   │ Chroma (vectors)    │◄────────┤  BM25 (keyword) │
   │ chunks.json (text)  │         └─────────────────┘
   └─────────────────────┘
```

## Stack

FastAPI · Chroma · sentence-transformers · rank-bm25 · pypdf · OpenCV · Tesseract · Streamlit

## Running it

Requires Python 3.12 and Tesseract (`brew install tesseract`).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Backend:
```bash
.venv/bin/uvicorn backend.main:app --reload
```

UI (second terminal):
```bash
.venv/bin/streamlit run ui/app.py
```

Then open http://localhost:8501. API docs are at http://localhost:8000/docs.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | liveness check |
| `GET` | `/stats` | index size, document list, chunk/vector consistency |
| `POST` | `/upload` | index a PDF / text file / image |
| `POST` | `/ask` | hybrid search (`method`, `alpha`, `k`) |
| `POST` | `/image-search` | OCR an image, search with the extracted text |
| `POST` | `/reset` | clear the index |

## Operational details

- Every response carries `X-Response-Time-ms`; anything over 1.5s logs at WARNING.
- `/stats` reports a `consistent` flag — if chunk count and vector count diverge, an ingest partially failed.
- `chunks.json` is written to a temp file and atomically renamed, so a crash mid-write can't corrupt it.
- Concurrent ingests are serialised with a lock to prevent lost writes.
- OCR below 70% average confidence logs a warning, so poor search results can be traced to an unreadable image rather than the ranker.
- Chunk IDs are content-hashed, so re-uploading the same file overwrites rather than duplicating.

## Learning path

`LEARN.md` walks through the whole thing from scratch — 7 runnable lessons in `lessons/`, each one building up to a piece of the final app.

## Known limits

- BM25 rebuilds its index whenever the corpus changes and scores every chunk, so keyword search cost grows linearly. Fine to ~100k chunks; past that it belongs in Elasticsearch/OpenSearch.
- Scanned PDFs with no text layer aren't OCR'd yet — pages would need rendering to images first.
- `/ask` returns the retrieved passages, not an LLM-generated answer. The prompt is assembled in `search.build_answer()` and just needs an LLM call to become full RAG.
