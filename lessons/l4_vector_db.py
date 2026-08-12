"""
LESSON 4 — Vector DB (Chroma)
Run me:  .venv/bin/python lessons/l4_vector_db.py

WHY A VECTOR DB EXISTS:
In lesson 3 you compared the query against every sentence, one by one.
Fine for 4 sentences. Death for 4 million.

A vector DB is just a database that stores those number-lists and can
find the closest ones FAST (it doesn't check all of them — it uses an
index called HNSW, basically a shortcut graph). Plus it stores the
original text and metadata alongside.

Normal DB : "give me the row where id = 5"          -> exact match
Vector DB : "give me the 5 rows closest in meaning" -> similarity match
"""

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# PersistentClient writes to disk, so the data survives a restart.
# (chromadb.Client() would be memory-only and vanish.)
client = chromadb.PersistentClient(path="data/lesson4_chroma")

# A "collection" is just a table.
# hnsw:space=cosine tells it to rank by cosine similarity (lesson 3).
collection = client.get_or_create_collection(
    name="demo",
    metadata={"hnsw:space": "cosine"},
)

docs = [
    "Refunds are processed within 7 working days.",
    "You can cancel your order before it ships.",
    "Our office is open Monday to Friday, 9am to 6pm.",
    "Delivery usually takes 3 to 5 days in metro cities.",
    "Error code SKU-99213 means the item is out of stock.",
]

# ---------------------------------------------------------------
# WRITE: every row needs an id, the text, its vector, and optional metadata.
# ids must be unique strings. Re-running this file just overwrites them.
# ---------------------------------------------------------------
collection.upsert(
    ids=[f"doc-{i}" for i in range(len(docs))],
    documents=docs,
    embeddings=model.encode(docs).tolist(),   # .tolist() -> chroma wants plain lists
    metadatas=[{"source": "faq.txt", "row": i} for i in range(len(docs))],
)

print("rows in collection:", collection.count())


# ---------------------------------------------------------------
# READ: embed the question the SAME way, ask for nearest neighbours.
# RULE: query and documents must use the SAME model. Mixing models = garbage.
# ---------------------------------------------------------------
question = "how long until my money comes back?"

res = collection.query(
    query_embeddings=[model.encode(question).tolist()],
    n_results=3,
)

print(f"\nQ: {question}")
for text, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
    # NOTE: chroma returns DISTANCE, not similarity. Lower = closer.
    # With cosine space: similarity = 1 - distance
    print(f"  sim={1 - dist:.3f}  [{meta['source']} row {meta['row']}]  {text}")


# ---------------------------------------------------------------
# Metadata filtering — combine "meaning" search with hard filters.
# This is what real products do: "search my notes, but only from last week".
# ---------------------------------------------------------------
res2 = collection.query(
    query_embeddings=[model.encode("when are you open?").tolist()],
    n_results=2,
    where={"row": {"$gte": 2}},        # only rows 2 and up
)
print("\nfiltered query:", res2["documents"][0])


# ---------------------------------------------------------------
# Watch it FAIL on purpose — this sets up lesson 5.
# ---------------------------------------------------------------
res3 = collection.query(
    query_embeddings=[model.encode("SKU-99213").tolist()],
    n_results=3,
)
print("\nexact-code search (vector only):")
for t, d in zip(res3["documents"][0], res3["distances"][0]):
    print(f"  sim={1 - d:.3f}  {t}")
print("\n^ it probably found it, but with a mediocre score, and other rows")
print("  crowded in. A keyword search would have been 100% certain. -> lesson 5")


# ---------------------------------------------------------------
# INTERVIEW NOTES
# ---------------------------------------------------------------
# Chroma  = easiest, embedded, great for demos and small apps.
# Qdrant  = production-grade, runs in Docker, better filtering + payloads.
# pgvector = a Postgres extension. Best pick when you ALREADY use Postgres
#            and don't want a second database to babysit.
# Pinecone / Weaviate / Milvus = managed / bigger cluster options.
#
# Good line: "I used Chroma because it's embedded and zero-ops for this scale.
#  If it grew, I'd move to Qdrant or pgvector — same code shape, better ops."
