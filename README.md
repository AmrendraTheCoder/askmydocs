# AskMyDocs

Upload documents or images, ask questions in plain English, get the exact passage back — using **hybrid retrieval** (BM25 keyword search fused with dense vector search) over a local vector database.

Small enough to read end to end in one sitting: ~1,000 lines of Python across six modules, with every design decision measured rather than asserted.

---

## What it does

- **Ingest** PDFs, text files, or images. Images go through an OpenCV cleanup pipeline and Tesseract OCR.
- **Chunk** documents by paragraph (not blind word windows) with overlap across boundaries.
- **Embed** each chunk with `all-MiniLM-L6-v2`, store the vectors in Chroma.
- **Search** with four selectable strategies — `rrf`, `weighted`, `vector`, `keyword` — so you can see exactly what each half of the hybrid contributes.
- **Search by photo** — upload a picture, its text is OCR'd and used as the query.
- **Measure** — `eval.py` scores retrieval against 32 labelled queries so claims here are numbers, not vibes.

## Why hybrid

The two retrievers fail on different things. BM25 matches tokens; vectors match meaning:

| Query | What BM25 sees | What vectors see |
|---|---|---|
| `"when do I get my money back"` | no shared words with "refund" — nothing to score | the paraphrase, correctly |
| `"AUTH-1203"` | an exact, rare token — its best case | a rare string with little semantic content |

Fusion is **Reciprocal Rank Fusion** by default (`1/(k + rank)`, k=60) — no score normalisation and no tuning needed. A normalised weighted blend is also implemented and selectable.

## Measured results

From `.venv/bin/python eval.py` — 32 labelled queries over a 31-chunk corpus, k=5. Each query is paired with a phrase that appears in exactly one place; a hit means that phrase came back in the top 5.

| method | paraphrase | exact code | mixed | recall@5 | MRR |
|---|---|---|---|---|---|
| keyword only | 58% | 100% | 100% | 84% | 0.768 |
| vector only | 83% | 100% | 88% | 91% | 0.787 |
| rrf (default) | 67% | 100% | 100% | 88% | 0.818 |
| **weighted** (α=0.5) | 83% | 100% | 100% | **94%** | **0.837** |

Two things worth saying out loud, because both cut against what I assumed when I built this:

**Keyword-only collapses on paraphrases — 58% against 83%.** This is the result that justifies the whole project. Ask a question in words the document doesn't use and BM25 has nothing to score.

**Vector-only got 100% on exact codes, not the near-misses I predicted.** An earlier version of this README claimed embeddings go fuzzy on identifiers like `AUTH-1203`. The eval says otherwise at this corpus size — 31 chunks is small enough that a rare token still lands in a distinctive region of the embedding space. I'd expect that advantage to erode as the corpus grows and near-duplicate codes crowd each other, but I haven't measured it, so it stays an expectation rather than a claim.

**RRF, the default, is not the best ranker here** — 88% against weighted's 94%. RRF only sees rank, so it throws away the information that BM25 scored one chunk far above the rest. It's the more robust default across unknown corpora, which is why it stays the default, but on this corpus weighted wins and the UI defaults to it.

Caveat worth keeping in mind: 32 queries over 31 chunks is a small benchmark. It's enough to show the paraphrase gap, not enough to rank the two fusion methods with confidence.

### Latency

Also from `eval.py`, plus end-to-end HTTP timing over 40 requests:

| | first request | p50 | p95 | max |
|---|---|---|---|---|
| before warmup + device fix | 12,464ms | 178ms | 644ms | 5,675ms |
| after | **58ms** | **7.9ms** | **17ms** | **28ms** |

Two separate bugs, both found by measuring rather than reading:

1. **The cold path was on the first user's request.** Loading the embedding model at startup wasn't enough — the first `/ask` still took 12.4s, because the BM25 index and Chroma's HNSW index are lazy too. `search.warmup()` runs one real search at boot so every lazy path is already hot.
2. **The model defaulted to Metal (`mps`), which is the wrong device here.** A search embeds one short string; the work is too small to amortise the GPU round trip. On CPU: p50 3.95ms vs 6.89ms, and p95 5.23ms vs 18.75ms. The tail matters more than the median — a p95 of 19ms and a max of 151ms is what makes a latency alert useless.

## Architecture

```
                    ┌──────────────┐
                    │  Browser UI  │   served by FastAPI itself
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

### On the UI

It's a single HTML file with no framework and no build step, served by FastAPI at `/`. That's a deliberate choice over the Streamlit prototype it replaced.

Streamlit is a fine way to put a face on a script, but it re-runs the whole file top-to-bottom on every interaction, keeps state in its own session object, and gives you almost no control over markup. It also meant running a second process on a second port, so "the app" was two things that could disagree about the API.

One `uvicorn` command now runs everything. The page talks to the same routes any other client would, so the API stays the only interface and the UI can't quietly drift from it. If this ever needs real client-side state, the honest next step is a small React or HTMX frontend — not a bigger pile of vanilla JS.

The results view draws a bar under each hit showing the keyword score against the vector score, so you can see which half of the hybrid actually found it. That's the project's whole thesis, made visible.

## Stack

FastAPI · Chroma · sentence-transformers · rank-bm25 · pypdf · OpenCV · Tesseract

## Running it

Requires Python 3.12 and Tesseract (`brew install tesseract`).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/uvicorn backend.main:app --reload
```

Open http://localhost:8000 for the UI. API docs are at http://localhost:8000/docs.

To reproduce the numbers above (this rebuilds the index from `data/eval_corpus/`):

```bash
.venv/bin/python eval.py
```

### Or with Docker, if you'd rather not install anything

```bash
docker build -t askmydocs . && docker run -p 7860:7860 askmydocs
```

Then open http://localhost:7860. The image bakes in the embedding model and
pre-seeds the demo index, so search works the moment it starts — no download
on first request and no empty-index demo. It's ~2.7GB, almost all of which is
PyTorch; that size is also why this can't run on a serverless host (see
`Dockerfile` for the full reasoning).

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | the UI |
| `GET` | `/health` | liveness check |
| `GET` | `/stats` | index size, document list, chunk/vector consistency |
| `POST` | `/upload` | index a PDF / text file / image |
| `POST` | `/ask` | hybrid search (`method`, `alpha`, `k`) |
| `POST` | `/image-search` | OCR an image, search with the extracted text |
| `POST` | `/reset` | clear the index |

## Operational details

- Every response carries `X-Response-Time-ms`; anything over 1.5s logs at WARNING.
- The search path is warmed at startup, so a slow `/ask` means a genuinely slow query rather than a cold cache.
- `/stats` reports a `consistent` flag — if chunk count and vector count diverge, an ingest partially failed.
- `chunks.json` is written to a temp file and atomically renamed, so a crash mid-write can't corrupt it.
- Concurrent ingests are serialised with a lock to prevent lost writes.
- OCR below 70% average confidence logs a warning, so poor search results can be traced to an unreadable image rather than the ranker.
- Chunk IDs are content-hashed, so re-uploading the same file overwrites rather than duplicating.
- Warmup failure is logged, not fatal — the service still boots and answers `/health` so an orchestrator can report "up but degraded".

## Known limits

- BM25 rebuilds its index whenever the corpus changes and scores every chunk, so keyword search cost grows linearly. Fine to ~100k chunks; past that it belongs in Elasticsearch/OpenSearch.
- The benchmark is 32 queries over 31 chunks. It's enough to show the paraphrase gap; it is not enough to rank `rrf` against `weighted` with confidence, and the labels are my own.
- Scanned PDFs with no text layer aren't OCR'd yet — pages would need rendering to images first.
- `/ask` returns the retrieved passages, not an LLM-generated answer. The prompt is assembled in `search.build_answer()` and just needs an LLM call to become full RAG.
- The whole index is single-process and file-backed. Two uvicorn workers would each hold their own BM25 cache and race on `chunks.json`.
