"""
Ingestion = take a file, turn it into searchable chunks.

THE PIPELINE:
    file -> raw text -> chunks -> embeddings -> stored

CHUNKING IS THE PART PEOPLE GET WRONG. Read this bit:

Why chunk at all? Two reasons.
  1. An embedding is ONE list of numbers. Squeeze a 40-page PDF into one
     embedding and it becomes a vague average of everything — it matches
     nothing well. Small pieces have sharp, specific meaning.
  2. You want to show the user the exact paragraph that answered them,
     not the whole document.

How big? ~150-250 words. Too small and a sentence loses its context.
Too big and the meaning blurs.

Why overlap? Because a fixed cut can slice a sentence — or worse, an
idea — in half. Repeating ~40 words at the boundary means the idea
survives whole in at least one chunk. Cheap insurance.
"""

import hashlib
import logging
import os
import re

from pypdf import PdfReader

from . import db, embedder, vision

log = logging.getLogger(__name__)

CHUNK_WORDS = 120       # target chunk size in words
OVERLAP_WORDS = 40      # how much context to repeat across a boundary

TEXT_EXT = {".txt", ".md", ".csv", ".log"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
PDF_EXT = {".pdf"}
SUPPORTED = TEXT_EXT | IMAGE_EXT | PDF_EXT


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Blind word-window chunking. The fallback for one giant wall of text."""
    words = text.split()
    if not words:
        return []

    chunks, start = [], 0
    step = size - overlap                      # how far the window slides
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Structure-aware chunking: respect paragraph boundaries first.

    WHY THIS BEATS A BLIND WORD WINDOW:
    A blank line is the author telling you "a new idea starts here". Cutting
    at word 200 regardless will slice a paragraph mid-thought, and that chunk
    now embeds as a blur of two half-ideas. Packing whole paragraphs up to
    the size limit keeps each chunk about ONE thing, which is exactly what
    makes its embedding sharp.

    Only when a single paragraph is itself bigger than the limit do we fall
    back to the blind window — at that point there's no structure to respect.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return _sliding_window(text, size, overlap)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > size:                     # oversized paragraph
            if current:
                chunks.append("\n\n".join(current))
                current, current_words = [], 0
            chunks.extend(_sliding_window(para, size, overlap))
            continue

        if current and current_words + para_words > size:
            chunks.append("\n\n".join(current))
            # Overlap = carry the last paragraph into the next chunk, so an
            # idea that spans a boundary survives whole somewhere.
            if len(current[-1].split()) <= overlap:
                current, current_words = [current[-1]], len(current[-1].split())
            else:
                current, current_words = [], 0

        current.append(para)
        current_words += para_words

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def read_pdf(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. Keeping the page number means you can
    tell the user 'this came from page 7' — a small touch that makes the
    app feel trustworthy."""
    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            log.exception("failed to extract page %d of %s", i, path)
            continue
        if text.strip():
            pages.append((i, text))

    if not pages:
        # A scanned PDF is just images in a PDF wrapper — no text layer at
        # all. This is THE most common "your app is broken" ticket for
        # document apps. Real fix: render pages to images and OCR them.
        log.warning("no extractable text in %s — probably a scanned PDF", path)
    return pages


def _make_id(source: str, page: int, index: int, text: str) -> str:
    """Deterministic id: same content re-uploaded overwrites itself instead
    of creating duplicates. Dedupe for free."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{source}::p{page}::c{index}::{digest}"


def ingest_file(path: str, source: str) -> dict:
    """Read one file, chunk it, embed it, store it. Returns a small report."""
    ext = os.path.splitext(source)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"unsupported file type '{ext}'. supported: {sorted(SUPPORTED)}")

    pages: list[tuple[int, str]] = []
    ocr_report = None

    if ext in PDF_EXT:
        pages = read_pdf(path)
    elif ext in IMAGE_EXT:
        ocr_report = vision.extract_text(path)
        if ocr_report["text"].strip():
            pages = [(1, ocr_report["text"])]
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            pages = [(1, f.read())]

    records = []
    for page_no, text in pages:
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "id": _make_id(source, page_no, i, chunk),
                "text": chunk,
                "source": source,
                "page": page_no,
            })

    if not records:
        log.warning("nothing indexable in %s", source)
        return {"source": source, "chunks": 0, "pages": len(pages),
                "ocr": ocr_report, "warning": "no readable text found in this file"}

    # Embed all chunks in ONE call — batching is the whole performance story.
    vectors = embedder.embed([r["text"] for r in records])
    db.add_chunks(records, vectors)

    return {"source": source, "chunks": len(records), "pages": len(pages), "ocr": ocr_report}
