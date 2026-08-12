"""
LESSON 7 — OpenCV + OCR (the "good to have" on the job ad)
Run me:  .venv/bin/python lessons/l7_opencv_ocr.py

This script MAKES its own test image, then reads the text back out of it,
so you don't need to find a sample file.

WHAT THESE TWO THINGS ARE:
  OpenCV (cv2) = a toolbox for images. Load, resize, blur, threshold,
                 find shapes. It is NOT AI. It's maths on pixels.
  OCR          = reading text out of an image. Tesseract does it.

THE ONE INSIGHT THAT MATTERS:
  OCR accuracy is mostly decided BEFORE tesseract runs. Clean the image
  first — grayscale, upscale, threshold — and accuracy jumps. That
  cleaning is the OpenCV part, and it's the part interviewers probe.
"""

import cv2
import numpy as np
import pytesseract

# ---------------------------------------------------------------
# 0. Make a messy test image (grey background, noise, slightly small text)
# ---------------------------------------------------------------
# Small text + low contrast + grain = a realistic bad phone scan.
np.random.seed(0)                                    # same image every run
img = np.full((100, 340, 3), 190, dtype=np.uint8)    # H x W x 3 (BGR!), grey
cv2.putText(img, "Invoice INV-2291", (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (95, 95, 95), 1)
cv2.putText(img, "Total: Rs 4,850.00", (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (95, 95, 95), 1)
noise = np.random.randint(0, 35, img.shape, dtype=np.uint8)
img = cv2.subtract(img, noise)                       # sprinkle grain on it
cv2.imwrite("data/lesson7_input.png", img)

print("image shape (height, width, channels):", img.shape)
print("NOTE: OpenCV loads colour as BGR, not RGB. Classic bug source.\n")


# ---------------------------------------------------------------
# 1. RAW OCR — no cleaning. Usually mediocre.
# ---------------------------------------------------------------
print("--- RAW (no preprocessing) ---")
print(repr(pytesseract.image_to_string(img).strip()))
print("^ look CLOSELY at the invoice number. It gets a digit wrong.")


# ---------------------------------------------------------------
# 2. THE PREPROCESSING PIPELINE. Learn these 5 steps by name.
# ---------------------------------------------------------------
def preprocess(bgr):
    # a) GRAYSCALE — colour tells OCR nothing. 3 channels -> 1. Faster too.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # b) UPSCALE — tesseract wants text roughly 30px tall. Small text = bad OCR.
    #    This single line is usually the biggest accuracy win. INTER_CUBIC
    #    guesses the in-between pixels smoothly instead of blockily.
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # c) DENOISE — median blur kills speckles without smearing edges
    #    (a normal blur would soften the letters; median keeps them crisp)
    gray = cv2.medianBlur(gray, 3)

    # d) THRESHOLD — force every pixel to pure black or pure white.
    #    Otsu picks the cut-off automatically instead of you guessing a number.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # e) (optional) DILATE — thickens strokes if letters came out broken
    # binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    return binary


clean = preprocess(img)
cv2.imwrite("data/lesson7_cleaned.png", clean)

print("\n--- CLEANED (grayscale -> upscale -> denoise -> Otsu threshold) ---")
print(repr(pytesseract.image_to_string(clean).strip()))
print("^ now the invoice number is correct. Same OCR engine. Same image.")
print("  The ONLY difference is 4 lines of OpenCV. That's the lesson.")
print("  (open data/lesson7_input.png and data/lesson7_cleaned.png to see it)")


# ---------------------------------------------------------------
# 3. OCR WITH CONFIDENCE + BOXES — this is how you filter garbage.
#    image_to_data gives you every word, where it is, and how sure it is.
#    Drop anything under ~60 confidence and your text gets much cleaner.
# ---------------------------------------------------------------
data = pytesseract.image_to_data(clean, output_type=pytesseract.Output.DICT)
print("\n--- per-word confidence ---")
kept = []
for word, conf in zip(data["text"], data["conf"]):
    word = word.strip()
    if not word:
        continue
    conf = float(conf)
    flag = "keep" if conf >= 60 else "DROP"
    print(f"  {conf:5.1f}  {flag}  {word}")
    if conf >= 60:
        kept.append(word)

print("\nfinal text:", " ".join(kept))


# ---------------------------------------------------------------
# 4. Bonus: --psm, the flag that fixes 80% of "OCR is bad" tickets.
#    psm = page segmentation mode = "what shape is this text?"
#      6  = one uniform block of text   (default-ish, good for documents)
#      7  = a single line               (good for a cropped field)
#      11 = sparse text anywhere        (good for screenshots / memes)
# ---------------------------------------------------------------
print("\npsm 11 (sparse text):", repr(pytesseract.image_to_string(clean, config="--psm 11").strip()))


print("""
==================================================================
INTERVIEW ANSWERS
==================================================================
Q: "How would you improve OCR accuracy?"
A: "Preprocess before OCR — grayscale, upscale to ~2x so glyphs are
    around 30px, median blur for noise, Otsu threshold to binarise, and
    deskew if the page is rotated. Then filter output by per-word
    confidence from image_to_data. If it's still bad I'd switch engine
    to PaddleOCR or EasyOCR, which beat Tesseract on photos and
    screenshots."

Q: "What's OpenCV actually doing here?"
A: "Pure pixel maths — no model. Colour conversion, resampling,
    a median filter, and Otsu's method picking a threshold by maximising
    between-class variance in the histogram."

Q: "Where else would you use OpenCV in this app?"
A: "Detect and crop the document region with contours before OCR, and
    deskew using minAreaRect on the text mask."
==================================================================
""")
