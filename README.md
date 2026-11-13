# RAG Chatbot (FastAPI + FAISS + Claude)

A tiny Retrieval-Augmented Generation chatbot in ~45 lines of Python. It answers questions using only the documents in `docs/`.

## How it works
1. **Load and chunk**: `docs/*.txt` are split into paragraphs.
2. **Embed and index**: chunks are embedded locally (`all-MiniLM-L6-v2`) and stored in a FAISS index.
3. **Retrieve**: the question is embedded and the top-3 most similar chunks are fetched.
4. **Generate**: Claude answers from those chunks only, and says "I don't know" otherwise (reduces hallucination).

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # then put your Anthropic API key in .env
uvicorn app:app --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI, or:
```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "What are your support hours?"}'
```

The response contains the `answer` and the retrieved `sources`.

## Ideas to extend
Add your own `.txt` files to `docs/`, persist the index with `faiss.write_index`, add chat history, stream responses, or swap in another LLM by editing the `llm.messages.create` call.
