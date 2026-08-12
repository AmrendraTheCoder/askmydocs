"""
LESSON 5 — BM25 = keyword search, the old boring one that still wins sometimes
Run me:  .venv/bin/python lessons/l5_bm25.py

WHAT BM25 IS, plain English:
It scores a document on "how many of the query's words are in it", but with
two bits of common sense bolted on:

  1. RARE WORDS COUNT MORE.  If you search "SKU-99213 stock", the word
     "stock" appears everywhere so it's nearly worthless. "SKU-99213"
     appears in one doc, so it's gold. (This part is called IDF.)

  2. REPEATING A WORD HAS DIMINISHING RETURNS, and long documents get
     penalised so they can't win just by being long.

That's the entire algorithm. It powers Elasticsearch, Lucene, most of
"normal" search on the internet. It has zero AI in it.
"""

from rank_bm25 import BM25Okapi

docs = [
    "Refunds are processed within 7 working days.",
    "You can cancel your order before it ships.",
    "Our office is open Monday to Friday, 9am to 6pm.",
    "Delivery usually takes 3 to 5 days in metro cities.",
    "Error code SKU-99213 means the item is out of stock.",
]


# ---------------------------------------------------------------
# TOKENIZING = chopping text into words. BM25 works on word lists,
# not raw strings. Keep it simple: lowercase + split.
# In the real app you'd also strip punctuation.
# ---------------------------------------------------------------
def tokenize(text: str):
    return text.lower().replace(".", " ").replace(",", " ").split()


bm25 = BM25Okapi([tokenize(d) for d in docs])


def run(query):
    scores = bm25.get_scores(tokenize(query))     # one score per document
    ranked = sorted(zip(scores, docs), reverse=True)
    print(f"\nquery: {query!r}")
    for score, doc in ranked[:3]:
        print(f"  {score:6.3f}  {doc}")


# WHERE BM25 IS BRILLIANT — exact tokens
run("SKU-99213")

# WHERE BM25 IS BRILLIANT — the literal word is present
run("refunds days")

# WHERE BM25 FALLS APART — no shared words at all.
# Vector search (lesson 3) nailed this one. BM25 scores everything ~0.
run("how long until my money comes back")


print("""
------------------------------------------------------------------
SO:
  BM25   wins on: exact terms, codes, names, rare words, typos-free queries
  Vector wins on: paraphrases, "I don't know the right word", meaning

  Neither is enough alone. Use BOTH. That's hybrid search. -> lesson 6
------------------------------------------------------------------
""")
