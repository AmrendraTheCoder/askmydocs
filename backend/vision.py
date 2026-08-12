"""
The computer-vision half: clean an image, then read the text out of it.
Everything here is lesson 7, tidied up for production use.
"""

import logging

import cv2
import numpy as np
import pytesseract

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 55.0     # tesseract's per-word confidence, 0-100


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Grayscale -> upscale -> denoise -> binarise.

    Order matters. Threshold LAST, because thresholding throws away all
    the grey information that denoising needs to do its job.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Only upscale small images. Blowing up a 4000px scan wastes seconds
    # of CPU per request and gains nothing — a real perf bug I'd expect
    # a support ticket about.
    if gray.shape[1] < 1200:
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten a tilted scan. Photos of paper are never square.

    Find every dark pixel, wrap the tightest possible rotated box around
    them all, read that box's angle, rotate the image back by it.
    """
    coords = np.column_stack(np.where(gray < 128))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:          # already straight, don't touch it
        return gray

    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def extract_text(image_path: str, psm: int = 3) -> dict:
    """Read an image file and return its text plus a quality report.

    psm 3  = automatic page segmentation. DEFAULT, and deliberately so —
             see "WHY THE LAYOUT SURVIVES" below.
    psm 6  = assume one uniform block. Better on a tight single-column
             crop, but it destroys paragraph structure.
    psm 11 = sparse text anywhere (screenshots, memes, UI).

    WHY THE LAYOUT SURVIVES
    ----------------------
    image_to_data() does not return a bag of words. Every word comes tagged
    with the block and paragraph it belongs to — Tesseract has already done
    the layout analysis. An earlier version of this function threw all of
    that away with `" ".join(words)`, producing one unbroken line.

    That mattered more than it looks. ingest.chunk_text() packs whole
    paragraphs and only falls back to a blind fixed-width window when it
    finds no blank lines. A flattened OCR string has no blank lines, ever,
    so EVERY image took the fallback path while typed documents got the
    good one.

    Two things were needed to fix it, and only one of them is obvious:

      1. Group words by (block_num, par_num) and join groups with a blank
         line, so chunk_text() can see the paragraph boundaries.
      2. Stop passing --psm 6. That flag means "assume a single uniform
         block", which explicitly tells Tesseract NOT to do the layout
         analysis whose output we then ask for. Measured on a 3-paragraph
         test image: psm 6 found 4 groups and glued each heading onto the
         previous body; psm 3 found all 6 cleanly.

    Note the grouping happens for every psm, including 11 — psm only
    changes how Tesseract segments the page, not whether it reports the
    segments. That is harmless for the query path in /image-search: the
    blank lines go through tokenize(), which keeps only [a-z0-9-]+ and
    drops whitespace either way.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"could not read image: {image_path}")

    clean = deskew(preprocess(bgr))

    data = pytesseract.image_to_data(
        clean,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )

    # image_to_data returns PARALLEL LISTS — index i in every list describes
    # the same word. That is why this walks an index instead of zipping.
    groups: dict[tuple[int, int], list[str]] = {}
    dropped, confidences = [], []
    word_count = 0

    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = float(data["conf"][i])
        if conf < MIN_CONFIDENCE:
            dropped.append(word)
            continue
        # dict preserves insertion order, so paragraphs come back in
        # reading order without an explicit sort.
        key = (data["block_num"][i], data["par_num"][i])
        groups.setdefault(key, []).append(word)
        confidences.append(conf)
        word_count += 1

    # A blank line between groups is exactly what chunk_text() splits on.
    text = "\n\n".join(" ".join(ws) for ws in groups.values())

    report = {
        "text": text,
        "word_count": word_count,
        "paragraphs": len(groups),
        "dropped_count": len(dropped),
        "avg_confidence": round(sum(confidences) / len(confidences), 1) if confidences else 0.0,
        "size": {"width": bgr.shape[1], "height": bgr.shape[0]},
    }

    # Low confidence is the #1 cause of "your search is broken" tickets.
    # Log it so future-you can prove the OCR was the problem, not the search.
    if report["avg_confidence"] < 70 and words:
        log.warning("low OCR confidence %.1f on %s — results may be poor",
                    report["avg_confidence"], image_path)

    return report
