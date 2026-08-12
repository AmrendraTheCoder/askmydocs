"""
The FastAPI app — all the HTTP routes.

Run it:   .venv/bin/uvicorn backend.main:app --reload
Docs at:  http://127.0.0.1:8000/docs
"""

import logging
import os
import shutil
import time

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import db, ingest, search, vision

# Structured-ish logs with timestamps. The first thing you want at 2am
# when someone says "it was slow around 3pm yesterday".
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("askmydocs")

app = FastAPI(
    title="AskMyDocs",
    version="1.0.0",
    description="Upload documents and images, search them with hybrid "
                "(BM25 + vector) retrieval.",
)

# Lets a browser frontend on another port call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your real domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = 25


# ---------------------------------------------------------------
# Middleware: time every request and log slow ones.
# This is your "how would you debug slow search?" answer, in code.
# ---------------------------------------------------------------
@app.middleware("http")
async def timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    level = logging.WARNING if elapsed_ms > 1500 else logging.INFO
    log.log(level, "%s %s -> %d in %.1fms",
            request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# ---------------------------------------------------------------
# request/response shapes
# ---------------------------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["what is the refund policy?"])
    k: int = Field(5, ge=1, le=20, description="how many results to return")
    alpha: float = Field(0.5, ge=0.0, le=1.0,
                         description="weighted mode only: 1.0=pure vector, 0.0=pure keyword")
    method: str = Field("rrf", pattern="^(rrf|weighted|keyword|vector)$")


# ---------------------------------------------------------------
# health + admin
# ---------------------------------------------------------------
@app.get("/health", tags=["support"])
def health():
    """Cheap liveness check. Never touch the model or DB here — a health
    endpoint that does real work will itself time out under load."""
    return {"ok": True}


@app.get("/stats", tags=["support"])
def stats():
    """Deeper check: is the index actually populated and consistent?
    If chunks != vectors, an ingest half-failed. That's the bug."""
    s = db.stats()
    s["consistent"] = s["chunks"] == s["vectors"]
    return s


@app.post("/reset", tags=["support"])
def reset():
    db.reset()
    return {"ok": True, "message": "index cleared"}


# ---------------------------------------------------------------
# ingest
# ---------------------------------------------------------------
@app.post("/upload", tags=["ingest"])
async def upload(file: UploadFile = File(...)):
    """Accept a PDF / txt / md / image, index it, return a report."""
    filename = os.path.basename(file.filename or "unnamed")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ingest.SUPPORTED:
        raise HTTPException(400, f"unsupported type '{ext}'. "
                                 f"allowed: {sorted(ingest.SUPPORTED)}")

    dest = os.path.join(db.UPLOAD_DIR, filename)
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    size_mb = os.path.getsize(dest) / 1_000_000
    if size_mb > MAX_UPLOAD_MB:
        os.remove(dest)
        raise HTTPException(413, f"file is {size_mb:.1f}MB, limit is {MAX_UPLOAD_MB}MB")

    try:
        report = ingest.ingest_file(dest, filename)
    except ValueError as e:
        # Bad input from the user -> 400, their problem, don't log as an error.
        raise HTTPException(400, str(e))
    except Exception as e:
        # Genuinely our bug -> log the full stack, return a clean message.
        # Never leak a stack trace to the caller.
        log.exception("ingest failed for %s", filename)
        raise HTTPException(500, f"could not index this file: {type(e).__name__}")

    return report


# ---------------------------------------------------------------
# search
# ---------------------------------------------------------------
@app.post("/ask", tags=["search"])
def ask(req: AskRequest):
    results = search.hybrid_search(req.query, k=req.k, alpha=req.alpha, method=req.method)
    return {
        "query": req.query,
        "method": req.method,
        "count": len(results),
        "results": results,
        "answer": search.build_answer(req.query, results),
    }


@app.post("/image-search", tags=["search"])
async def image_search(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=20),
    psm: int = Query(11, description="6 = block of text, 11 = sparse/screenshot"),
):
    """Upload an image, OCR it, then search the index with the text found.

    Snap a photo of a paragraph -> find the related passages in your docs.
    """
    filename = os.path.basename(file.filename or "query.png")
    if os.path.splitext(filename)[1].lower() not in ingest.IMAGE_EXT:
        raise HTTPException(400, "please upload an image file")

    dest = os.path.join(db.UPLOAD_DIR, f"_query_{filename}")
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    try:
        ocr = vision.extract_text(dest, psm=psm)
    except Exception:
        log.exception("OCR failed for %s", filename)
        raise HTTPException(500, "could not read this image")

    if not ocr["text"].strip():
        return {"ocr": ocr, "count": 0, "results": [],
                "warning": "no text found in the image — try psm=6 for documents"}

    results = search.hybrid_search(ocr["text"], k=k)
    return {"ocr": ocr, "extracted_query": ocr["text"],
            "count": len(results), "results": results}
