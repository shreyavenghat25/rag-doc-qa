# RAG Document Q&A

Ask questions about your own PDFs, web pages, or text and get answers grounded in the source, with inline citations back to the exact file and page.

The retrieval pipeline combines lexical (BM25) and semantic (dense vector) search, fuses them with Reciprocal Rank Fusion, and reranks the candidates with a cross-encoder before the LLM writes an answer. Answers stream token by token over Server-Sent Events.

**Live demo: [shreya-rag-docs.streamlit.app](https://shreya-rag-docs.streamlit.app)** — click "Try a sample" in the sidebar to ask questions about this README, or upload your own PDF.

## Architecture

```
Documents (PDF / URL / Text)
        │
        ▼
  ┌─────────────┐
  │  Ingestion  │  PyMuPDF · BeautifulSoup
  └──────┬──────┘
         │ pages
         ▼
  ┌─────────────┐
  │   Chunking  │  Semantic (cosine-similarity breakpoints) · recursive fallback
  └──────┬──────┘
         │ chunks
         ▼
  ┌─────────────┐
  │  Embedding  │  BAAI/bge-small-en-v1.5  (384-dim, normalized, runs locally)
  └──────┬──────┘
         │ vectors
    ┌────┴──────┐
    │  Storage  │
    │  FAISS    │  HNSWFlat  M=16  efConstruction=200
    │  SQLite   │  chunk text + metadata
    └────┬──────┘
         │
         ▼  at query time
  ┌──────────────────────────────────┐
  │          Hybrid Retrieval        │
  │  BM25 (rank_bm25)                │
  │  FAISS dense search              │
  │  Reciprocal Rank Fusion  k=60    │
  │  Cross-encoder reranking         │  ms-marco-MiniLM-L-6-v2
  └──────────────┬───────────────────┘
                 │ top-5 chunks
                 ▼
  ┌─────────────────────────────────┐
  │  Generation  (Groq, gpt-oss-20b)│  [Source N] inline citations
  └─────────────────────────────────┘
                 │
     {answer, citations, latency_ms}   (JSON or SSE stream)
```

## Quickstart

Requires Python 3.11 and a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys).

```bash
git clone https://github.com/shreyavenghat25/rag-doc-qa
cd rag-doc-qa

cp .env.example .env          # then set GROQ_API_KEY in .env
pip install -r requirements.txt

uvicorn app.main:app --port 8000        # API  → http://localhost:8000/docs
streamlit run frontend/app.py           # UI   → http://localhost:8501  (second terminal)
```

Self-contained mode (no API server; the UI runs the pipeline in-process, as the hosted demo does):

```bash
RAG_MODE=embedded streamlit run frontend/app.py
```

Or with Docker:

```bash
docker compose up --build
```

API at http://localhost:8000/docs, UI at http://localhost:8501, MLflow at http://localhost:5000.

## API Endpoints

| Method | Path                   | Description                |
| ------ | ---------------------- | -------------------------- |
| POST   | `/api/v1/upload`       | Upload and index a PDF     |
| POST   | `/api/v1/index/url`    | Index a web page           |
| POST   | `/api/v1/index/text`   | Index plain text           |
| POST   | `/api/v1/query`        | Ask a question (JSON)      |
| GET    | `/api/v1/query/stream` | Ask a question (SSE stream)|
| GET    | `/api/v1/documents`    | List indexed documents     |
| GET    | `/api/v1/health`       | Health check               |

## Key Design Decisions

**Hybrid retrieval.** BM25 catches exact terms (names, codes, acronyms) that embeddings blur; dense search catches paraphrases that share no words with the query. Running both covers each one's blind spot.

**Reciprocal Rank Fusion (k=60).** BM25 scores and cosine similarities live on different scales, so adding them requires tuning weights. RRF combines the two rankings by position only, which needs no score normalization.

**Cross-encoder reranking.** The bi-encoder embeds query and chunk separately, which is fast but approximate. The cross-encoder reads each (query, chunk) pair together and scores relevance more precisely, so it is applied only to the ~20 fused candidates.

**Semantic chunking.** Chunks are split where the cosine similarity between consecutive sentences drops, so related sentences stay together instead of being cut at a fixed character count.

**Non-blocking streaming.** The SSE endpoint runs retrieval and the blocking LLM stream in a worker thread, so one long answer does not stall other requests on the event loop.

**Two deployment modes, one pipeline.** The Streamlit UI talks to a backend interface with two implementations: an HTTP client for the FastAPI service (local and Docker), and an embedded mode that calls the same pipeline in-process (the hosted demo, where only one process is available). Retrieval, reranking, and generation code is identical in both.

**Per-visitor isolation in the demo.** The API keeps one persistent index. A public demo can't, or every visitor would search everyone else's uploads. In embedded mode each browser session gets a private in-memory FAISS index and chunk store, while the embedding model and reranker are loaded once and shared. Re-uploading the same file is detected by content hash and skipped, and sessions are capped at 5 documents to keep memory bounded on a free host.

**Citation mapping.** The model is prompted to cite `[Source N]`; the server parses those markers (tolerating `[N]` and `[Source 1, 3]` variants) and maps them back to the retrieved chunk, file, and page.

## Evaluation

An evaluation harness (`app/eval/ragas_eval.py`) computes RAGAS faithfulness, answer relevancy, context recall, and context precision and logs runs to MLflow. A benchmark question set and results comparing retrieval configurations (dense-only, BM25-only, hybrid, hybrid + reranker) are in progress and will be published here.

## Project Structure

```
rag-doc-qa/
├── app/
│   ├── api/routes.py          # FastAPI endpoints (JSON + SSE)
│   ├── core/
│   │   ├── ingestion.py       # PDF / URL / text loaders
│   │   ├── chunker.py         # Semantic + recursive chunking
│   │   ├── embedder.py        # Sentence-transformer wrapper
│   │   ├── vector_store.py    # FAISS HNSWFlat index
│   │   ├── retriever.py       # BM25 + dense + RRF + reranker
│   │   ├── generator.py       # Prompt, Groq call, citation parsing
│   │   └── indexing.py        # Ingestion orchestration
│   ├── eval/ragas_eval.py     # RAGAS evaluation + MLflow logging
│   ├── database.py            # SQLite metadata store
│   ├── config.py              # Settings via pydantic-settings
│   └── main.py                # FastAPI app
├── app/core/session.py        # Per-visitor in-memory index (demo mode)
├── frontend/
│   ├── app.py                 # Streamlit UI
│   ├── backends.py            # API client / embedded backend
│   └── requirements.txt       # Lean CPU-only deps for the hosted demo
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Tech Stack

Python · FastAPI · Streamlit · sentence-transformers · FAISS · rank-bm25 · Groq (gpt-oss-20b) · RAGAS · MLflow · SQLite · Docker

## Demo

Uploading a PDF and asking for a summary on the [live demo](https://shreya-rag-docs.streamlit.app). Each point cites the chunk it came from:

![Answer with inline citations](screenshots/demo1.png)

![Sources and latency](screenshots/demo2.png)
