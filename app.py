"""Minimal RAG chatbot: FastAPI + FAISS + sentence-transformers + Claude."""
import glob

import anthropic
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()  # loads ANTHROPIC_API_KEY from the .env file
app = FastAPI(title="RAG Chatbot")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # local embedding model
llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment (set by .env)

# 1. Load documents and split them into paragraph-sized chunks
chunks = [
    p.strip()
    for path in glob.glob("docs/*.txt")
    for p in open(path, encoding="utf-8").read().split("\n\n")
    if p.strip()
]

# 2. Embed the chunks and store them in a FAISS index (inner product == cosine on normalized vectors)
vectors = embedder.encode(chunks, normalize_embeddings=True)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)


class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(body: Question, k: int = 3):
    # 3. Retrieve the k most similar chunks
    query = embedder.encode([body.question], normalize_embeddings=True)
    _, ids = index.search(query, min(k, len(chunks)))
    sources = [chunks[i] for i in ids[0]]

    # 4. Generate an answer grounded in the retrieved context
    context = "\n\n".join(sources)
    try:
        reply = llm.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=500,
            system="Answer using only the provided context. If the answer is not there, say you don't know.",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.question}"}],
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e.message}")
    return {"answer": reply.content[0].text, "sources": sources}
