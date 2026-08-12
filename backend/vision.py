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


def extract_text(image_path: str, psm: int = 6) -> dict:
    """Read an image file and return its text plus a quality report.

    psm 6  = one uniform block (documents, invoices)
    psm 11 = sparse text anywhere (screenshots, memes, UI)
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

    words, dropped, confidences = [], [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        if not word:
            continue
        conf = float(conf)
        if conf >= MIN_CONFIDENCE:
            words.append(word)
            confidences.append(conf)
        else:
            dropped.append(word)

    text = " ".join(words)
    report = {
        "text": text,
        "word_count": len(words),
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
