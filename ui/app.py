"""
Streamlit UI. Run it:  .venv/bin/streamlit run ui/app.py
(the backend must already be running on port 8000)

HOW STREAMLIT WORKS — the one thing that confuses everyone:
This whole file re-runs TOP TO BOTTOM every single time you click anything.
There is no onClick handler, no state, no re-render diffing. Click a button,
the script runs again from line 1, and `st.button(...)` returns True on that
one run.

That's why anything expensive must be cached (@st.cache_data) and anything
that must survive a click must live in st.session_state.
"""

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")


def api_up() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=2).json().get("ok", False)
    except requests.RequestException:
        return False


# ---------------------------------------------------------------
# Sidebar — search controls + the support/admin panel
# ---------------------------------------------------------------
with st.sidebar:
    st.title("📄 AskMyDocs")

    if not api_up():
        st.error("Backend is down.\n\nStart it with:\n\n"
                 "`.venv/bin/uvicorn backend.main:app --reload`")
        st.stop()          # nothing below this line runs
    st.success("Backend healthy")

    st.subheader("Search settings")
    method = st.selectbox(
        "Fusion method", ["rrf", "weighted", "vector", "keyword"],
        help="rrf = rank fusion (default). Switch to vector/keyword to see "
             "each half on its own — great for demoing why hybrid wins.",
    )
    alpha = st.slider("alpha (weighted only)", 0.0, 1.0, 0.5, 0.05,
                      help="1.0 = pure meaning, 0.0 = pure keywords")
    top_k = st.slider("results", 1, 10, 5)

    st.divider()
    st.subheader("Index status")
    stats = requests.get(f"{API}/stats", timeout=5).json()
    c1, c2 = st.columns(2)
    c1.metric("chunks", stats["chunks"])
    c2.metric("documents", stats["documents"])
    if not stats["consistent"]:
        st.warning(f"chunks ({stats['chunks']}) != vectors ({stats['vectors']}) "
                   "— an ingest half-failed")
    if stats["sources"]:
        st.caption("Indexed: " + ", ".join(stats["sources"]))

    if st.button("Clear index", type="secondary"):
        requests.post(f"{API}/reset", timeout=10)
        st.rerun()


# ---------------------------------------------------------------
# Main area
# ---------------------------------------------------------------
tab_ask, tab_upload, tab_image = st.tabs(["🔍 Ask", "📤 Upload", "🖼️ Image search"])


def render_results(results):
    if not results:
        st.info("No matches. Upload a document first, or try different words.")
        return
    for r in results:
        header = f"#{r['rank']}  ·  {r['source']} p{r['page']}  ·  score {r['score']:.4f}"
        with st.expander(header, expanded=(r["rank"] == 1)):
            st.write(r["text"])
            a, b = st.columns(2)
            a.caption(f"keyword (BM25): {r['keyword_score']:.3f}")
            b.caption(f"vector (cosine): {r['vector_score']:.3f}")


with tab_ask:
    st.subheader("Ask your documents")
    query = st.text_input("Question", placeholder="when do I get my money back?")

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Searching..."):
            resp = requests.post(
                f"{API}/ask",
                json={"query": query, "k": top_k, "alpha": alpha, "method": method},
                timeout=60,
            )
        if resp.status_code != 200:
            st.error(f"{resp.status_code}: {resp.text}")
        else:
            data = resp.json()
            st.caption(f"{data['count']} results · method: {data['method']}")
            render_results(data["results"])


with tab_upload:
    st.subheader("Add documents")
    st.caption("PDF, txt, md, csv, log, or an image (png/jpg — text is read via OCR)")

    uploaded = st.file_uploader("Choose a file",
                                type=["pdf", "txt", "md", "csv", "log",
                                      "png", "jpg", "jpeg", "webp"])
    if uploaded and st.button("Index it", type="primary"):
        with st.spinner("Reading, chunking, embedding..."):
            resp = requests.post(
                f"{API}/upload",
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=300,
            )
        if resp.status_code != 200:
            st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
        else:
            report = resp.json()
            if report.get("warning"):
                st.warning(report["warning"])
            else:
                st.success(f"Indexed {report['chunks']} chunks "
                           f"from {report['pages']} page(s)")
            if report.get("ocr"):
                ocr = report["ocr"]
                st.caption(f"OCR: {ocr['word_count']} words kept, "
                           f"{ocr['dropped_count']} dropped, "
                           f"avg confidence {ocr['avg_confidence']}")
            st.rerun()


with tab_image:
    st.subheader("Search by photo")
    st.caption("Upload a photo or screenshot. Its text is read with "
               "OpenCV + Tesseract, then used as the search query.")

    img = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"],
                           key="imgsearch")
    psm = st.radio("What kind of image?",
                   [("6", "A document / block of text"),
                    ("11", "A screenshot / scattered text")],
                   format_func=lambda x: x[1], horizontal=True)

    if img and st.button("Read & search", type="primary"):
        left, right = st.columns([1, 2])
        left.image(img, caption="query image", use_container_width=True)
        with st.spinner("Running OCR..."):
            resp = requests.post(
                f"{API}/image-search?k={top_k}&psm={psm[0]}",
                files={"file": (img.name, img.getvalue())},
                timeout=120,
            )
        with right:
            if resp.status_code != 200:
                st.error(f"{resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                ocr = data["ocr"]
                st.info(f"Text found ({ocr['word_count']} words, "
                        f"avg confidence {ocr['avg_confidence']}):\n\n"
                        f"{ocr['text'][:400] or '— nothing —'}")
                if data.get("warning"):
                    st.warning(data["warning"])
                render_results(data["results"])
