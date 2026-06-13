"""
Indexing pipeline — orchestrates the full ingestion flow:
load → chunk → embed → store in FAISS + SQLite
"""
import numpy as np
from app.core.ingestion import load_pdf, load_url, load_text
from app.core.chunker import recursive_character_split, semantic_chunk
from app.core.embedder import get_embedder
from app.core.vector_store import get_faiss_index
from app.database import init_db, insert_document, insert_chunks
from app.config import settings


def index_pdf(file_bytes: bytes, filename: str, use_semantic_chunking: bool = True) -> dict:
    """Index a PDF file. Returns indexing stats."""
    init_db()
    pages = load_pdf(file_bytes, filename)
    return _index_pages(pages, filename, "pdf", use_semantic_chunking)


def index_url(url: str, use_semantic_chunking: bool = True) -> dict:
    """Index a URL. Returns indexing stats."""
    init_db()
    pages = load_url(url)
    return _index_pages(pages, url, "url", use_semantic_chunking)


def index_text(text: str, filename: str = "plain_text", use_semantic_chunking: bool = False) -> dict:
    """Index plain text. Returns indexing stats."""
    init_db()
    pages = load_text(text, filename)
    return _index_pages(pages, filename, "text", use_semantic_chunking)


def _index_pages(pages: list[dict], source: str, source_type: str, use_semantic: bool) -> dict:
    embedder = get_embedder()
    faiss_index = get_faiss_index(embedder.dimension)

    # Chunking
    if use_semantic and len(pages) > 0:
        chunks = semantic_chunk(pages, embedder)
        if len(chunks) < 3:  # fallback if too few chunks
            chunks = recursive_character_split(pages)
    else:
        chunks = recursive_character_split(pages)

    if not chunks:
        return {"error": "No text extracted from document", "chunks": 0}

    print(f"[Indexing] {source}: {len(pages)} pages → {len(chunks)} chunks")

    # Embed
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_batch(texts)

    # Add to FAISS
    faiss_ids = faiss_index.add(vectors)
    faiss_index.save()

    # Save to SQLite
    doc_id = insert_document(source, source_type)
    chunk_records = [
        {
            "chunk_index": i,
            "faiss_id": faiss_ids[i],
            "text": chunks[i]["text"],
            "metadata": chunks[i].get("metadata", {})
        }
        for i in range(len(chunks))
    ]
    insert_chunks(doc_id, chunk_records)

    # Refresh BM25 index
    from app.core.retriever import get_retriever
    get_retriever().refresh_bm25()

    return {
        "doc_id": doc_id,
        "source": source,
        "source_type": source_type,
        "pages": len(pages),
        "chunks": len(chunks),
        "faiss_total": faiss_index.size,
        "chunking_strategy": "semantic" if use_semantic else "recursive"
    }
