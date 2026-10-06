"""
FastAPI routes:
POST /upload        — upload PDF
POST /index/url     — index a URL
POST /index/text    — index plain text
POST /query         — query the RAG pipeline
GET  /query/stream  — SSE streaming query
GET  /documents     — list indexed documents
GET  /health        — health check
"""
import json
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl

from app.core.indexing import index_pdf, index_url, index_text
from app.core.retriever import get_retriever
from app.core.generator import generate_answer, extract_citations
from app.database import list_documents

router = APIRouter()


# ─── Request/Response Models ────────────────────────────────────────────────

class URLIndexRequest(BaseModel):
    url: str
    use_semantic_chunking: bool = True


class TextIndexRequest(BaseModel):
    text: str
    filename: str = "plain_text"
    use_semantic_chunking: bool = False


class QueryRequest(BaseModel):
    query: str
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    use_reranker: bool = True
    stream: bool = False


# ─── Ingestion Endpoints ────────────────────────────────────────────────────

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    use_semantic_chunking: bool = True
):
    """Upload and index a PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    result = index_pdf(file_bytes, file.filename, use_semantic_chunking)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {"status": "indexed", **result}


@router.post("/index/url")
async def index_from_url(request: URLIndexRequest):
    """Fetch and index content from a URL."""
    try:
        result = index_url(str(request.url), request.use_semantic_chunking)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {"status": "indexed", **result}


@router.post("/index/text")
async def index_from_text(request: TextIndexRequest):
    """Index plain text directly."""
    if len(request.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short (min 50 chars)")

    result = index_text(request.text, request.filename, request.use_semantic_chunking)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {"status": "indexed", **result}


# ─── Query Endpoints ─────────────────────────────────────────────────────────

@router.post("/query")
async def query(request: QueryRequest):
    """
    Query the RAG pipeline.
    Returns {answer, citations, latency_ms, tokens_used}.
    """
    retriever = get_retriever()
    chunks = retriever.retrieve(
        query=request.query,
        top_k_retrieve=request.top_k_retrieve,
        top_k_rerank=request.top_k_rerank,
        use_reranker=request.use_reranker
    )

    if not chunks:
        return {
            "answer": "No relevant documents found. Please upload documents first.",
            "citations": [],
            "latency_ms": 0,
            "tokens_used": 0,
            "chunks_retrieved": 0
        }

    result = generate_answer(request.query, chunks, stream=False)
    return result


@router.get("/query/stream")
def query_stream(
    query: str,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5
):
    """
    SSE streaming query endpoint.
    Events: {"type": "token"} ... {"type": "citations"} then [DONE].

    Declared with plain `def` (not async) so FastAPI runs retrieval in a
    worker thread, and StreamingResponse iterates the sync generator in a
    threadpool, so the blocking Groq stream never stalls the event loop.
    """
    retriever = get_retriever()
    chunks = retriever.retrieve(query=query, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank)

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    if not chunks:
        def no_docs():
            yield sse({"type": "token", "text": "No relevant documents found."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_docs(), media_type="text/event-stream")

    result = generate_answer(query, chunks, stream=True)

    def event_stream():
        full_text = ""
        start = time.perf_counter()
        try:
            for token in result["stream"]:
                full_text += token
                yield sse({"type": "token", "text": token})
        except Exception as e:
            yield sse({"type": "error", "text": str(e)})

        yield sse({
            "type": "citations",
            "citations": extract_citations(full_text, result["chunks"]),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Utility Endpoints ────────────────────────────────────────────────────────

@router.get("/documents")
async def get_documents():
    """List all indexed documents."""
    docs = list_documents()
    return {"documents": docs, "total": len(docs)}


@router.get("/health")
async def health():
    from app.core.vector_store import get_faiss_index
    faiss_idx = get_faiss_index()
    return {
        "status": "ok",
        "faiss_vectors": faiss_idx.size,
    }
