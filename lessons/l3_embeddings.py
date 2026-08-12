"""
LESSON 3 — Embeddings. The single most important idea in the whole job ad.
Run me:  .venv/bin/python lessons/l3_embeddings.py
(first run downloads a ~90MB model, takes a minute — after that it's instant)

THE IDEA IN ONE LINE:
An embedding turns a sentence into a list of numbers, arranged so that
sentences with similar MEANING end up with similar numbers.

That's it. That's the whole trick. Everything else is bookkeeping.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# all-MiniLM-L6-v2 = small, fast, free, runs on your laptop, no API key.
# It is THE default answer when someone asks "which embedding model?"
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat",
    "A kitten is resting on the rug",     # different words, SAME meaning
    "Python is a programming language",
    "I deployed the API to production",
]

vectors = model.encode(sentences)     # -> shape (4, 384)

print("shape:", vectors.shape)
print("=> 4 sentences, each became 384 numbers\n")
print("first 8 numbers of sentence 1:", np.round(vectors[0][:8], 3), "\n")


# ---------------------------------------------------------------
# How do we compare two of these? COSINE SIMILARITY.
# Plain English: are these two arrows pointing the same direction?
#   1.0  = identical meaning
#   0.0  = unrelated
#  -1.0  = opposite
# ---------------------------------------------------------------
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    # np.dot        = multiply pairwise and add up
    # np.linalg.norm = length of the arrow
    # dividing by the lengths removes "how long", leaves "which direction"


print("cat/mat  vs  kitten/rug   :", round(cosine(vectors[0], vectors[1]), 3), " <- high, no shared words!")
print("cat/mat  vs  python lang  :", round(cosine(vectors[0], vectors[2]), 3), " <- low")
print("python   vs  deployed API :", round(cosine(vectors[2], vectors[3]), 3), " <- middling, both tech")


# ---------------------------------------------------------------
# Now the actual magic: SEMANTIC SEARCH in 5 lines.
# Note the query shares ZERO words with the winning sentence.
# ---------------------------------------------------------------
query = "a small animal is relaxing"
qv = model.encode(query)

scores = [(cosine(qv, v), s) for v, s in zip(vectors, sentences)]
scores.sort(reverse=True)     # highest first

print(f"\nquery: {query!r}")
for score, sentence in scores:
    print(f"  {score:.3f}  {sentence}")


# ---------------------------------------------------------------
# WHERE THIS BREAKS  (say this in the interview, it sounds senior)
# ---------------------------------------------------------------
# Vector search is bad at EXACT things: product codes, order IDs, names,
# error codes. Ask for "SKU-99213" and it may return "SKU-99127" because
# they "look similar in meaning". A dumb keyword search would nail it.
#
# That is exactly WHY hybrid search exists. -> Lesson 5 and 6.
