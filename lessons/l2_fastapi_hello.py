"""
LESSON 2 — FastAPI = Express.js for Python
Run me:  .venv/bin/uvicorn lessons.l2_fastapi_hello:app --reload --port 8001
Then open:  http://127.0.0.1:8001/docs     <-- THE KILLER FEATURE

That /docs page is auto-generated. You never write it. You can click
"Try it out" and fire real requests. Show this in the interview.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Lesson 2 API")      # JS: const app = express()

# a fake in-memory database
NOTES = {1: "buy milk", 2: "learn fastapi"}


# ---------------------------------------------------------------
# GET  — same as app.get("/", ...) in express
# ---------------------------------------------------------------
@app.get("/")                 # the @ thing is a "decorator" = a wrapper.
def root():                   # it just means "this function handles GET /"
    return {"message": "hello"}     # return a dict -> FastAPI turns it into JSON


# ---------------------------------------------------------------
# Path parameter — the {note_id} lands in the function argument.
# Because you typed it as `int`, FastAPI validates it for you.
# Try /notes/abc in the browser -> you get a clean 422 error, free.
# ---------------------------------------------------------------
@app.get("/notes/{note_id}")
def get_note(note_id: int):
    if note_id not in NOTES:
        raise HTTPException(status_code=404, detail="note not found")
    return {"id": note_id, "text": NOTES[note_id]}


# ---------------------------------------------------------------
# Query parameter — /search?q=milk&limit=5
# Anything not in the path becomes a query param automatically.
# ---------------------------------------------------------------
@app.get("/search")
def search(q: str, limit: int = 10):        # limit has a default -> optional
    hits = [t for t in NOTES.values() if q.lower() in t.lower()]
    return {"query": q, "results": hits[:limit]}


# ---------------------------------------------------------------
# POST with a body — this is where Pydantic shines.
# A "Pydantic model" is basically a TypeScript interface that is REAL
# at runtime: it validates the incoming JSON and rejects bad data for you.
# ---------------------------------------------------------------
class NoteIn(BaseModel):
    text: str
    pinned: bool = False        # optional, defaults to False


@app.post("/notes")
def create_note(note: NoteIn):          # <- FastAPI reads the body into this
    new_id = max(NOTES) + 1
    NOTES[new_id] = note.text
    return {"id": new_id, "pinned": note.pinned}


# ---------------------------------------------------------------
# THE DEV-SUPPORT ENDPOINT. Always add this. Say this in the interview.
# Monitoring tools ping /health every 30s to know if your service is alive.
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "notes_count": len(NOTES)}


# ---------------------------------------------------------------
# WHAT TO NOTICE
# ---------------------------------------------------------------
# 1. You never wrote validation. The type hints did it.
# 2. You never wrote API docs. /docs did it.
# 3. `uvicorn file:app --reload` is basically `nodemon server.js`.
#    uvicorn = the server that runs your app (like the http server under express)
