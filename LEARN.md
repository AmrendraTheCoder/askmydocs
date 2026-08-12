# LEARN.md — your actual one-day course

Everything here is already installed and already working on your laptop. You don't have to set anything up. You just have to **run things and read what comes out.**

Read this like a WhatsApp chat. Every big word gets explained the moment it shows up.

---

## Before anything: the one command you need

Open Terminal, then:

```bash
cd ~/Desktop/askmydocs
```

Everything below assumes you're in that folder.

You'll see `.venv/bin/python` a lot. That's not magic — `.venv` is just a folder holding this project's own private copy of Python and its packages. It's `node_modules`, basically. Using `.venv/bin/python` instead of plain `python` means "use THIS project's stuff". That's the whole idea.

---

## The map (what you're about to learn, in order)

```
  Lesson 1   Python, if you already know JavaScript
  Lesson 2   FastAPI — building a backend
  Lesson 3   Embeddings — turning meaning into numbers   ← the core idea
  Lesson 4   Vector DB — storing and searching those numbers
  Lesson 5   BM25 — old-school keyword search
  Lesson 6   HYBRID — mixing 4 and 5                     ← the job ad, literally
  Lesson 7   OpenCV + OCR — reading text out of images
  --------
  The app    all 7 lessons glued together into a real product
```

Do them in order. Each one takes 10–20 minutes of actually reading the output and poking at it.

---

## Lesson 1 — Python for a Node person

```bash
.venv/bin/python lessons/l1_python_for_node_devs.py
```

Then **open the file and read it.** The output is boring; the comments are the lesson.

The only three things that will trip you up coming from JS:

1. **Indentation is the braces.** 4 spaces. If your indentation is wrong, your code means something different. That's it, that's the whole "Python is weird" thing.
2. **`obj["key"]`, never `obj.key`** for dictionaries.
3. **List comprehension** — `[x*2 for x in nums]` is just `nums.map(x => x*2)` written inside-out. You'll see it in every AI codebase.

There's a full JS→Python cheat map at the bottom of that file. That's your reference sheet.

**Time: 20 min. Don't over-study this.** You'll absorb the rest by reading the project code.

---

## Lesson 2 — FastAPI (this is Express, renamed)

```bash
.venv/bin/uvicorn lessons.l2_fastapi_hello:app --reload --port 8001
```

Then open **http://127.0.0.1:8001/docs** in your browser.

That page is auto-generated. You did not write it. You can click "Try it out" on any endpoint and fire a real request from the browser.

**Stop and appreciate this**, because it's an interview line: in Express you'd install Swagger, write a YAML spec, keep it in sync, and it'd still drift. FastAPI reads your Python type hints and generates it, always correct.

The pieces:

| Thing | What it is |
|---|---|
| `@app.get("/x")` | The `@` line is a **decorator** — a wrapper. It means "this function handles GET /x". Same as `app.get("/x", handler)`. |
| `def f(note_id: int)` | The `: int` isn't decoration. FastAPI **validates** it. Send `abc` and you get a clean 422 error for free. |
| `class NoteIn(BaseModel)` | A **Pydantic model**. A TypeScript interface that actually exists at runtime and rejects bad JSON before your code sees it. |
| `uvicorn` | The server that runs the app. `uvicorn main:app --reload` ≈ `nodemon server.js`. |

Stop the server with `Ctrl+C` when you're done.

**Time: 30 min.** Play with `/docs`. Break things on purpose — send a string where an int goes and watch the error.

---

## Lesson 3 — Embeddings ⭐ THE BIG ONE

```bash
.venv/bin/python lessons/l3_embeddings.py
```

**The whole idea in one sentence:** an embedding turns a sentence into a list of numbers, arranged so that sentences with similar *meaning* end up with similar numbers.

Look at what it prints:

```
cat/mat  vs  kitten/rug   : 0.603   ← high, and they share ZERO words
cat/mat  vs  python lang  : 0.031   ← low
```

"The cat sat on the mat" and "A kitten is resting on the rug" have no words in common. The computer still knows they mean the same thing. **That is the entire AI-search revolution.** Everything else is plumbing around this one fact.

Two words to own:

- **Embedding** = the list of numbers. Ours is 384 numbers long.
- **Cosine similarity** = how you compare two of them. Think of each embedding as an arrow pointing somewhere in space. Cosine asks: *are these two arrows pointing the same direction?* `1.0` = identical, `0.0` = unrelated.

**Now the important half.** Scroll to the bottom of that file. Vector search is *bad* at exact things — product codes, order IDs, names. Ask it for `SKU-99213` and it might hand you `SKU-99127`, because to it they "look similar in meaning".

Remember that failure. It's the reason lesson 6 exists.

**Time: 20 min.** Edit the `sentences` list, add your own, re-run, see what it scores.

---

## Lesson 4 — Vector database

```bash
.venv/bin/python lessons/l4_vector_db.py
```

In lesson 3 you compared the query against every sentence one by one. Fine for 4. Impossible for 4 million.

**A vector DB is just a database that stores those number-lists and finds the closest ones fast.** That's the whole product category.

| | Normal DB (Postgres) | Vector DB (Chroma) |
|---|---|---|
| Question it answers | "the row where id = 5" | "the 5 rows closest in meaning" |
| Match type | exact | similarity |

Three things to notice in the output:

1. **It returns `distance`, not similarity.** Lower = closer. With cosine space, `similarity = 1 - distance`. Getting this backwards is a genuinely common bug — your search returns the *worst* results and looks broken.
2. **Metadata filtering** — you can say "search by meaning, but only rows from last week". Real products live on this.
3. **The last query fails on purpose.** Searching `SKU-99213` gives a mediocre score with junk crowding in. Setting up lesson 5.

**The names, so you can drop them:** Chroma (easy, embedded, what we use), Qdrant (production, runs in Docker), pgvector (a Postgres extension — the smart pick if you already have Postgres and don't want a second database to babysit), Pinecone/Weaviate/Milvus (managed/bigger).

**Time: 20 min.**

---

## Lesson 5 — BM25, the boring one that still wins

```bash
.venv/bin/python lessons/l5_bm25.py
```

BM25 is keyword search. Zero AI. It counts how many of your query's words appear in each document, with two bits of common sense:

1. **Rare words count more.** Search "SKU-99213 stock" — "stock" is everywhere so it's nearly worthless; "SKU-99213" is in one document, so it's gold.
2. **Long documents get penalised**, so they can't win just by being long and containing everything.

That's the algorithm. It powers Elasticsearch and most of "normal" internet search.

Look at the output:

```
query: 'SKU-99213'                      →  nails it, 1.046, everything else 0.000
query: 'how long until my money back'   →  0.000, 0.000, 0.000  ← totally blind
```

**So now you have two tools with exactly opposite strengths:**

| | good at | blind to |
|---|---|---|
| BM25 | exact terms, codes, names | paraphrases |
| Vectors | meaning, paraphrases | exact codes |

Neither is enough. Which is the entire point of the next lesson.

**Time: 15 min.**

---

## Lesson 6 — HYBRID SEARCH ⭐⭐ THIS IS THE JOB AD

```bash
.venv/bin/python lessons/l6_hybrid.py
```

The job ad said **"Hybrid & Vector DB"**. This file is that line.

**The problem:** BM25 spits out scores like `0.0, 3.7, 12.4` (no fixed range). Vectors give `0.31, 0.88` (always -1 to 1). You can't just add them — BM25 would bully the vector score into irrelevance.

So you **fuse** them. Two ways, and you should know both:

### Method 1 — weighted fusion
Squash both score lists into 0→1 (**min-max normalisation**: `(x - min) / (max - min)`), then blend:

```
final = alpha × vector + (1 - alpha) × keyword
```

`alpha` = how much you trust meaning over keywords. 0.5 is a fine default.

*Good:* explainable, tunable. *Bad:* one weird outlier score wrecks the normalisation.

### Method 2 — Reciprocal Rank Fusion (RRF)
Throw the scores away. Use only the **position** in each list:

```
score = 1 / (60 + rank)     ... added up across both lists
```

Rank 1 → 1/61. Rank 2 → 1/62. Add them up.

*Good:* no normalisation, no tuning, immune to crazy score ranges. This is what Elasticsearch and Qdrant ship as their **default**. Saying "I used RRF" reads as senior, because most candidates have never heard of it.

Look at the three queries in the output. Watch the `kw` and `vec` columns:

- `"when do I get my money back"` → keyword column is `0.00`, vector carries it
- `"SKU-99213"` → both hit, keyword is certain
- `"cancel order delivery time"` → both contribute

**One retriever is blind on each query. The other covers it. That's hybrid, and that's why it's on the job ad.**

The exact sentences to say in the interview are at the bottom of that file. Read them twice.

**Time: 40 min. This is the lesson worth over-studying.**

---

## Lesson 7 — OpenCV + OCR

```bash
.venv/bin/python lessons/l7_opencv_ocr.py
```

- **OpenCV** = a toolbox for images. Load, resize, blur, threshold. **It is not AI.** It's maths on pixels.
- **OCR** = reading text out of an image. Tesseract does that.

The script builds a deliberately bad scan, then reads it twice. Look at the output:

```
RAW      : 'Invoice INV-2201'   ← wrong digit
CLEANED  : 'Invoice INV-2291'   ← correct
```

**Same image. Same OCR engine. The only difference is 4 lines of OpenCV.** In a real product that wrong digit is a support ticket: "your app read my invoice number wrong."

The 4 lines, in order — learn these by name:

1. **Grayscale** — colour tells OCR nothing, and 3 channels → 1 is faster.
2. **Upscale (3x, INTER_CUBIC)** — Tesseract wants text ~30px tall. Usually the single biggest win.
3. **Median blur** — kills speckle noise without smearing the letter edges (a normal blur would soften the strokes; median keeps them crisp).
4. **Otsu threshold** — force every pixel to pure black or white. "Otsu" just means it picks the cut-off automatically instead of you guessing a number.

Then two extras that matter:

- **Confidence filtering** — `image_to_data` gives you a 0–100 confidence per word. Drop anything under ~60 and your text gets dramatically cleaner.
- **`--psm`** — "page segmentation mode", i.e. *what shape is this text?* `6` = a block of text (documents), `11` = scattered text (screenshots, memes). Switching this fixes most "OCR is bad" complaints.

Open `data/lesson7_input.png` and `data/lesson7_cleaned.png` side by side to see what the cleaning actually did.

**Time: 30 min.**

---

## Now: the real app

Everything above, glued together. Two terminal windows.

**Terminal 1 — the backend:**
```bash
cd ~/Desktop/askmydocs && .venv/bin/uvicorn backend.main:app --reload
```

**Terminal 2 — the UI:**
```bash
cd ~/Desktop/askmydocs && .venv/bin/streamlit run ui/app.py
```

Then open **http://localhost:8501**.

Try this exact sequence — it's also your interview demo script:

1. **Upload** tab → upload `data/sample_faq.txt`
2. **Ask** tab → *"what happens if the courier already picked up my parcel"* → the CANCELLATION passage comes first, even though your question shares almost no words with it
3. Sidebar → switch **Fusion method** to `keyword`, ask the same thing → watch it get worse
4. Switch to `vector`, ask `AUTH-1203` → watch *that* get worse
5. Switch back to `rrf` → both work

**Step 3 and 4 are the demo.** You're not just showing a working app, you're showing you know *why* the architecture is what it is. That's the difference between a candidate and a hire.

### What each file does

| File | Job | Lesson it came from |
|---|---|---|
| `backend/embedder.py` | loads the model once, turns text → numbers | 3 |
| `backend/db.py` | Chroma (vectors) + chunks.json (raw text for BM25) | 4 |
| `backend/ingest.py` | file → text → **chunks** → embeddings → stored | 3, 4 |
| `backend/search.py` | **the hybrid ranker** — the heart of it | 5, 6 |
| `backend/vision.py` | clean image → OCR → text | 7 |
| `backend/main.py` | the HTTP routes | 2 |
| `ui/app.py` | the Streamlit UI | — |

**Read `backend/search.py` twice.** If you understand only one file in this project, make it that one.

### The one concept in the app that wasn't in a lesson: chunking

**Chunking = cutting a document into small pieces before embedding.** Why:

1. An embedding is *one* list of numbers. Squeeze a 40-page PDF into one and it becomes a vague average of everything — it matches nothing well. Small pieces have sharp, specific meaning.
2. You want to show the user the exact paragraph that answered them, not the whole PDF.

Size ~120–250 words. **Overlap** ~40 words repeated across each boundary, so an idea that straddles a cut survives whole in at least one chunk.

And the upgrade that's worth mentioning: this project chunks **by paragraph**, not by blind word count. A blank line is the author telling you "new idea starts here". Cutting at word 200 regardless slices a paragraph mid-thought, and that chunk now embeds as a blur of two half-ideas. See `chunk_text()` in `ingest.py`.

---

## The "dev support" half of the job — it's already in the code

They said **"dev + dev support"**. Half building, half keeping the built thing alive and answering "why is it broken?"

Most candidates have nothing to say about the second half. You do — point at these:

| In the code | What you say |
|---|---|
| `GET /health` (`main.py`) | "Cheap liveness check for monitoring. Deliberately touches nothing — a health check that does real work will itself time out under load." |
| `GET /stats` + `consistent` flag | "Deeper check. If `chunks != vectors`, an ingest half-failed. That's the bug, found in one request." |
| The timing middleware | "Every response carries `X-Response-Time-ms`, and anything over 1.5s logs at WARNING. That's my answer to 'search feels slow' — I don't guess, I look." |
| `try/except` split in `/upload` | "Bad input from the user → 400 and no error log. Our bug → 500, full stack trace in the logs, clean message to the caller. **Never leak a stack trace to the client.**" |
| `os.replace()` in `db.py` | "Write to a temp file, then rename. Rename is atomic, so a crash mid-write leaves the old good file instead of half a file." |
| `threading.Lock` in `db.py` | "Two uploads at once would both read-modify-write the same file and one would lose. Classic race condition." |
| Low-OCR-confidence warning | "When OCR confidence is under 70 I log a warning. So when someone says 'search is broken', I can show the search was fine — the image was unreadable." |

**And when they ask "user says search is slow, what do you check?"** — walk the pipeline out loud:

> "First I'd check the response-time header and logs to see *which* endpoint. Then I'd time the three stages separately — embedding the query, the vector lookup, and the BM25 scoring. My guess would be BM25, because it rebuilds its index whenever the corpus changes and it scores every chunk in the collection, so it grows linearly. The vector side doesn't, it's indexed. Fix would be caching the index — which I do — and past ~100k chunks moving keyword search to Elasticsearch."

That answer is worth more than any amount of "yes I know Python".

---

## Jargon → one-line translation

| Word | It means |
|---|---|
| Embedding | a list of numbers that captures the meaning of text |
| Cosine similarity | are these two arrows pointing the same way? 1 = yes, 0 = unrelated |
| Vector DB | a database that finds the closest number-lists, fast |
| Semantic search | search by meaning, not exact words |
| BM25 | keyword search that weights rare words higher |
| Hybrid search | keyword + semantic, fused into one ranking |
| RRF | fuse two rankings using position, not score. `1/(60+rank)` |
| Normalisation | squashing different score ranges into 0→1 so they're comparable |
| Chunking | cutting a document into small pieces before embedding |
| RAG | find relevant chunks → hand them + the question to an LLM → answer |
| Reranker | a slower, smarter model that re-sorts your top ~50 results |
| HNSW | the index a vector DB uses so it doesn't compare against everything |
| OCR | reading text out of an image |
| Otsu threshold | auto-picking the black/white cut-off point |
| psm | telling Tesseract what *shape* the text is |
| Endpoint | one URL your API answers on |
| Pydantic | Python library that validates request bodies from type hints |
| venv | a per-project folder of Python packages. `node_modules`. |

---

## If you only have 3 hours

1. Lesson 3 (embeddings) — 20 min
2. Lesson 5 (BM25) — 15 min
3. **Lesson 6 (hybrid) — 45 min. Non-negotiable.**
4. Run the app, do the 5-step demo above — 30 min
5. Read `backend/search.py` line by line — 30 min
6. Read the "dev support" table above — 20 min

Skip lessons 1, 2, 4, 7 if you must. **Never skip 6.**

---

## The two traps

1. **Trying to understand everything before building anything.** You'll burn 3 hours on Python syntax and never open `search.py`. The code is already running — read it *while* it runs.
2. **Being able to run it but not explain it.** They will ask *why*. Why hybrid? Why chunk? Why does the vector DB return distance? If you can answer "why", nobody cares whether you memorised the syntax.

---

## The honest summary

You already know backend. Python is the same shapes with different syntax. A vector DB is a database that stores number arrays. Hybrid search is two searches added together. OpenCV is maths on pixels. Streamlit is a shortcut around HTML.

There is no magic here. There's one genuinely clever idea — that meaning can be a list of numbers — and everything else is ordinary engineering around it.

Run the lessons. Read `search.py`. Do the 5-step demo until it's smooth.

You're fine. 🚀
