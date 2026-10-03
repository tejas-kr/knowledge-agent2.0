# Personal Knowledge Base Agent

A retrieval-augmented generation (RAG) system over a folder of research PDFs. It loads documents, chunks them, generates embeddings with a local Ollama model, stores the vectors in ChromaDB, and answers questions by retrieving the most similar chunks and grounding a local LLM in that context.

## Pipeline

```
PDFs (documents/) -> load -> chunk (500 words, 100 overlap) -> embed (Ollama)
                    -> store (ChromaDB, cosine distance) -> retrieve (top-k)
                    -> prompt -> generate (Ollama) -> answer with [S#] citations
```

## Requirements

- Python 3.14, [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally (default `http://localhost:11434`)
- Node.js 18+ and npm (for the React UI)

## Setup

```bash
uv sync

ollama pull nomic-embed-text-v2-moe:latest   # embedding model
ollama pull qwen3:1.7b                       # generation model
```

Put your PDFs in `documents/`.

## Usage

```bash
uv run knowledge-agent2-0                        # interactive REPL
uv run knowledge-agent2-0 -q "your question"     # ask one question, then exit
uv run knowledge-agent2-0 --top-k 3              # change number of retrieved chunks
uv run knowledge-agent2-0 --force                # re-chunk and re-embed everything
```

The vector store is persisted in `data/chroma/`. Re-runs skip embedding if the store is already populated. Set `OLLAMA_HOST` to use a remote Ollama server.

## API

Run the FastAPI server:

```bash
uv run knowledge-agent2-0-api              # http://127.0.0.1:8000
# or: uv run uvicorn knowledge_agent2_0.api:app --reload
```

| Method | Endpoint | Body | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | — | Status and vector store size |
| `POST` | `/ingest` | `{"force": false}` | Chunk, embed, and store documents |
| `POST` | `/retrieve` | `{"question": "...", "top_k": 5}` | Similar chunks only, no generation |
| `POST` | `/ask` | `{"question": "...", "top_k": 5}` | Retrieve + generate, returns answer and sources |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

Ingest documents once (via the CLI or `POST /ingest`) before asking; `/ask` and `/retrieve` return `503` while the vector store is empty.

Example response from `/ask`:

```json
{
  "answer": "Retrieval-augmented generation (RAG) is a technique ... [S1]",
  "sources": [
    {
      "chunk_id": "2410.12837v1_p13_c1",
      "document_id": "2410.12837v1",
      "filename": "2410.12837v1.pdf",
      "page_number": 13,
      "chunk_index": 1,
      "score": 0.6422,
      "content": "boundaries of what is possible in retrieval-augmented AI applications."
    }
  ]
}
```

## UI

A React chat interface lives in `ui/` (Vite, React 19):

```bash
cd ui
npm install
npm run dev          # http://localhost:5173
```

Run the API server first — the Vite dev server proxies `/health`, `/ingest`, `/retrieve`, and `/ask` to `http://localhost:8000`.

The UI provides:

- Status bar with indexed chunk count and an **Ingest documents** button (with force-rebuild toggle)
- Chat-style question input with adjustable Top-K
- Answers with expandable source cards showing filename, page, and similarity score

Production build:

```bash
npm run build        # outputs ui/dist/
```

## Project structure

```
knowledge-agent2.0/
├── documents/                  # source PDFs
├── data/chroma/                # persisted ChromaDB vector store (gitignored)
├── ui/                         # React (Vite) chat interface
│   ├── src/
│   │   ├── App.jsx             # chat UI, sources, ingest controls
│   │   ├── api.js              # fetch wrappers for the API
│   │   └── styles.css
│   ├── vite.config.js          # dev proxy -> localhost:8000
│   └── package.json
├── src/knowledge_agent2_0/
│   ├── config.py               # models, chunk size/overlap, top_k, paths
│   ├── ollama_client.py        # shared Ollama client
│   ├── main.py                 # CLI entrypoint
│   ├── api.py                  # FastAPI app (ingest/retrieve/ask)
│   ├── ingest/
│   │   ├── loader.py           # PDF -> pages (pymupdf, page metadata)
│   │   ├── chunker.py          # word-based chunking with overlap
│   │   ├── embedder.py         # Ollama embeddings (batched) + query embedding
│   │   └── pipeline.py         # load -> chunk -> embed -> store
│   ├── store/
│   │   ├── vectorstore.py      # ChromaDB persistent client/collection
│   │   └── similarity.py       # cosine similarity utility
│   ├── retrieve/
│   │   ├── retriever.py        # semantic search over the vector store
│   │   └── prompt.py           # system prompt + context/user prompt builder
│   └── generate/
│       ├── llm.py              # chat completion via Ollama
│       └── pipeline.py         # retrieve -> generate
└── tests/                      # pytest suite (35 tests)
```

## Configuration

All settings live in `src/knowledge_agent2_0/config.py`:

| Setting | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `nomic-embed-text-v2-moe:latest` | Ollama embedding model |
| `GENERATION_MODEL` | `qwen3:1.7b` | Ollama generation model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Word-based chunking |
| `TOP_K` | `5` | Chunks retrieved per question |
| `COLLECTION_NAME` | `research-papers` | ChromaDB collection |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server (env var) |

## Tests

```bash
uv run pytest -q
```

Tests cover chunking, cosine similarity, the ChromaDB store, retrieval ranking, the full ingest-to-retrieve pipeline, and the FastAPI endpoints (including `422`/`502`/`503` error paths). Tests use ephemeral in-memory collections with fake embedders, so no Ollama is required.

## Notes

- The knowledge base is stored as vectors in ChromaDB with cosine distance (`score = 1 - distance`), so scores behave like cosine similarity in `[0, 1]` for normalized embeddings.
- Chunk IDs (`{document}_p{page}_c{index}`) make re-ingestion idempotent via upserts.
- Answers are grounded only in the retrieved context; the system prompt instructs the model to cite `[S#]` source labels and to say when context is insufficient.
