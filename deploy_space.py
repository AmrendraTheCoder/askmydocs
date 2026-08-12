"""
Deploy this repo to a Hugging Face Space (Docker SDK).

WHY HF SPACES AND NOT VERCEL/NETLIFY — the short version, because it's the
first thing anyone asks:

  torch alone unpacks to 635MB, against a 250MB serverless function limit.
  Tesseract is a system binary, not a pip package.
  Chroma and chunks.json need a disk that survives between requests.

Any one of those rules out serverless. Spaces runs the container built by
the Dockerfile next to this file, with 2 vCPU / 16GB RAM on the free tier.

Usage:
    export HF_TOKEN=hf_...          # https://huggingface.co/settings/tokens ("write")
    python deploy_space.py          # optionally: --user NAME --space NAME

The Space README needs YAML front-matter for Spaces to configure itself.
That front-matter would look like junk at the top of the GitHub README, so
it's generated here and uploaded only to the Space.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

TITLE = "AskMyDocs"
EMOJI = "🔍"
SHORT = ("Hybrid document search — BM25 fused with dense vectors. "
         "Ask in plain English, get the exact passage back.")

FRONT_MATTER = """---
title: {title}
emoji: {emoji}
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: {short}
---

# {title}

{short}

Hybrid retrieval: BM25 keyword search fused with dense vectors. Measured at
**94% recall@5** over 32 labelled queries — keyword-only collapses to **58%**
on paraphrased questions, which is the whole argument for the hybrid.

The demo index is pre-seeded with a small support-docs corpus, so search works
the moment the Space boots. Upload your own PDFs, text files or images to add
to it. Each result shows a bar of its keyword score against its vector score,
so you can see which half of the hybrid found it.

Source, eval harness and full write-up: https://github.com/AmrendraTheCoder/askmydocs
"""

# Everything the image builds for itself, plus local cruft. The Space builds
# from the Dockerfile, so shipping a local index or venv would only bloat the
# upload and then be overwritten at build time anyway.
IGNORE = [
    ".venv/*", "**/__pycache__/*", "*.pyc", ".git/*", ".claude/*",
    "data/chroma/*", "data/lesson4_chroma/*", "data/uploads/*",
    "data/chunks.json", "data/eval_results.json", ".DS_Store",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("HF_USER", "AmrendraTheCoder"))
    ap.add_argument("--space", default=os.environ.get("HF_SPACE", "askmydocs"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.\n"
              "  1. https://huggingface.co/settings/tokens -> New token -> type 'write'\n"
              "  2. export HF_TOKEN=hf_...\n"
              "  3. re-run this script", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    repo_id = f"{args.user}/{args.space}"
    api = HfApi(token=token)

    print(f"creating space {repo_id} (docker sdk)...")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)

    root = Path(__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(FRONT_MATTER.format(title=TITLE, emoji=EMOJI, short=SHORT))
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space")

    print("uploading source...")
    api.upload_folder(folder_path=str(root), repo_id=repo_id, repo_type="space",
                      ignore_patterns=IGNORE + ["README.md", "deploy_space.py"])

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\ndone: {url}")
    print("first build takes ~8-12 minutes (torch + model bake). "
          "Watch the Logs tab; it's ready when it says 'Application startup complete'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
