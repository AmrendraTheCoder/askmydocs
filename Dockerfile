# AskMyDocs — container image for Hugging Face Spaces (or any Docker host).
#
# Why not Vercel/Netlify: this needs ~1GB of dependencies (torch alone is
# 529MB), a Tesseract system binary, and a writable disk for Chroma. Serverless
# function platforms give you none of those. Anything that runs a container
# with a few hundred MB of RAM will run this.

FROM python:3.12-slim

# tesseract-ocr is a system binary, not a pip package — this is the line that
# makes serverless platforms impossible for this app.
# No libgl1 needed: requirements pin opencv-python-headless, which is the
# right build for a server that never opens a window.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as UID 1000. Everything the app writes at runtime
# (chroma/, chunks.json, uploads/) has to be owned by that user or the first
# upload fails with a permission error that looks like an app bug.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# CPU-only torch on purpose. The default wheel bundles CUDA and unpacks to
# ~2.5GB for a GPU this container will never have. The CPU wheel is ~200MB,
# and embedder.py pins the model to CPU anyway because it measured faster
# for single-query inference.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user \
        torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . .

# Bake the embedding model into the image. Downloading it on first boot would
# make the first visitor wait ~12s and would make the Space fail entirely if
# huggingface.co were unreachable at start-up.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

# Pre-seed the demo index. A search box over an empty index is a broken
# demo — the visitor types a question, gets nothing, and leaves.
RUN python -c "\
import os; \
from backend import db, ingest; \
db.reset(); \
d='data/eval_corpus'; \
[ingest.ingest_file(os.path.join(d,f), f) for f in sorted(os.listdir(d)) if f.endswith('.txt')]; \
print('seeded:', db.stats())"

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
