"""
In-memory RAG session — one per browser session in the hosted demo.

The FastAPI service keeps a single persistent index (FAISS on disk + SQLite).
That's wrong for a public demo: every visitor would search everyone else's
uploads. A RAGSession gives each visitor a private, in-memory FAISS index and
chunk store, while the heavy models (embedder, reranker) stay shared and are
loaded once per process.
"""
import hashlib
import time
from typing import Iterator

from app.core.embedder import get_embedder
from app.core.generator import extract_citations, generate_answer
from app.core.indexing import chunk_pages
from app.core.ingestion import load_pdf, load_text, load_url
from app.core.retriever import HybridRetriever
from app.core.vector_store import FAISSIndex

MAX_DOCUMENTS = 5
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB keeps memory predictable on a free host


class InMemoryStore:
    """Same read interface as app.database, backed by plain Python lists."""

    def __init__(self):
        self._chunks: list[dict] = []
        self._by_faiss_id: dict[int, dict] = {}
        self.documents: list[dict] = []

    def add_document(self, filename: str, source_type: str, chunks: list[dict], faiss_ids: list[int]) -> int:
        doc_id = len(self.documents) + 1
        self.documents.append({
            "id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "total_chunks": len(chunks),
        })
        for i, (chunk, faiss_id) in enumerate(zip(chunks, faiss_ids)):
            record = {
                "id": len(self._chunks) + 1,
                "doc_id": doc_id,
                "chunk_index": i,
                "faiss_id": faiss_id,
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "filename": filename,
                "source_type": source_type,
            }
            self._chunks.append(record)
            self._by_faiss_id[faiss_id] = record
        return doc_id

    def get_all_chunks(self) -> list[dict]:
        return [dict(c) for c in self._chunks]

    def get_chunks_by_faiss_ids(self, faiss_ids: list[int]) -> list[dict]:
        return [dict(self._by_faiss_id[i]) for i in faiss_ids if i in self._by_faiss_id]


class RAGSession:
    def __init__(self):
        self.embedder = get_embedder()  # shared, loaded once per process
        self.index = FAISSIndex(dimension=self.embedder.dimension, persist=False)
        self.store = InMemoryStore()
        self.retriever = HybridRetriever(self.embedder, self.index, store=self.store)
        self._hashes: set[str] = set()

    # ── Indexing ──────────────────────────────────────────────────────────────
    def index_pdf(self, file_bytes: bytes, filename: str, use_semantic: bool = True) -> dict:
        if len(file_bytes) > MAX_PDF_BYTES:
            raise ValueError(f"PDF too large for the demo (max {MAX_PDF_BYTES // (1024 * 1024)} MB)")
        return self._index(load_pdf(file_bytes, filename), filename, "pdf", use_semantic, file_bytes)

    def index_url(self, url: str, use_semantic: bool = True) -> dict:
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return self._index(load_url(url), url, "url", use_semantic, url.encode())

    def index_text(self, text: str, filename: str = "pasted_text", use_semantic: bool = False) -> dict:
        return self._index(load_text(text, filename), filename, "text", use_semantic, text.encode())

    def _index(self, pages: list[dict], source: str, source_type: str, use_semantic: bool, raw: bytes) -> dict:
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self._hashes:
            return {"status": "duplicate", "source": source, "chunks": 0}
        if len(self.store.documents) >= MAX_DOCUMENTS:
            raise ValueError(f"Demo limit: {MAX_DOCUMENTS} documents per session. Refresh the page to start over.")

        chunks = chunk_pages(pages, self.embedder, use_semantic)
        if not chunks:
            raise ValueError("No text could be extracted from this document")

        vectors = self.embedder.embed_batch([c["text"] for c in chunks])
        faiss_ids = self.index.add(vectors)
        self.store.add_document(source, source_type, chunks, faiss_ids)
        self.retriever.refresh_bm25()
        self._hashes.add(digest)
        return {"status": "indexed", "source": source, "pages": len(pages), "chunks": len(chunks)}

    def list_documents(self) -> list[dict]:
        return list(reversed(self.store.documents))

    # ── Querying ──────────────────────────────────────────────────────────────
    def _retrieve(self, query: str, top_k_retrieve: int, top_k_rerank: int, use_reranker: bool = True):
        return self.retriever.retrieve(
            query=query,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            use_reranker=use_reranker,
        )

    def query(self, query: str, top_k_retrieve: int = 20, top_k_rerank: int = 5, use_reranker: bool = True) -> dict:
        chunks = self._retrieve(query, top_k_retrieve, top_k_rerank, use_reranker)
        if not chunks:
            return {
                "answer": "No relevant documents found. Please add a document first.",
                "citations": [], "latency_ms": 0, "tokens_used": 0, "chunks_retrieved": 0,
            }
        return generate_answer(query, chunks, stream=False)

    def query_stream(self, query: str, top_k_retrieve: int = 20, top_k_rerank: int = 5) -> Iterator[dict]:
        """Yields the same events as the API's SSE endpoint: token… then citations."""
        start = time.perf_counter()
        chunks = self._retrieve(query, top_k_retrieve, top_k_rerank)
        if not chunks:
            yield {"type": "token", "text": "No relevant documents found. Please add a document first."}
            return
        full_text = ""
        try:
            for token in generate_answer(query, chunks, stream=True)["stream"]:
                full_text += token
                yield {"type": "token", "text": token}
        except Exception as e:  # surface API failures in the UI instead of crashing the app
            yield {"type": "error", "text": str(e)}
        yield {
            "type": "citations",
            "citations": extract_citations(full_text, chunks),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
