"""
Answer generation with Groq.

generate_answer(stream=False) -> {answer, citations, latency_ms, tokens_used, chunks_retrieved}
generate_answer(stream=True)  -> {stream: Iterator[str], chunks: list[dict]}
"""
import re
import time
from typing import Iterator

from groq import Groq

from app.config import settings

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context chunks.
- Cite every factual claim inline, immediately after the sentence it supports, using the exact format [Source N].
  Example: "The model is trained on 10,000 samples [Source 2]."
- Never use bare [N] markers, and never collect all citations in a list at the end.
- If the answer is not in the context, say "I don't have enough context to answer this."
- Be concise but complete.
"""

# Accepts [Source 1], [Source 1, 3], [1], [1, 2]: models don't always follow the format exactly.
_CITATION_RE = re.compile(r"\[(?:Source\s*)?(\d+(?:\s*,\s*(?:Source\s*)?\d+)*)\]", re.IGNORECASE)
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (local) or Space secrets (deployed).")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _source_name(chunk: dict) -> str:
    return chunk.get("filename") or chunk.get("metadata", {}).get("source", "Unknown")


def build_context_prompt(query: str, chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("metadata", {}).get("page", "?")
        parts.append(f"[Source {i}] (from: {_source_name(chunk)}, page {page})\n{chunk['text']}")
    context = "\n---\n".join(parts)
    return f"Context:\n\n{context}\n\nQuestion: {query}"


def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_context_prompt(query, chunks)},
    ]


def _llm_kwargs() -> dict:
    kwargs = {
        "model": settings.llm_model,
        "max_tokens": settings.max_tokens,
        "temperature": 0.1,
    }
    # gpt-oss models reason before answering; keep it short for latency.
    if settings.llm_model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return kwargs


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Map [Source N] markers in the answer back to the retrieved chunks."""
    cited = {
        int(n)
        for group in _CITATION_RE.findall(answer)
        for n in re.findall(r"\d+", group)
    }
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        if i in cited:
            citations.append({
                "source_n": i,
                "chunk_id": chunk.get("id"),
                "faiss_id": chunk.get("faiss_id"),
                "filename": _source_name(chunk),
                "page": chunk.get("metadata", {}).get("page"),
                "text_preview": chunk["text"][:200],
                "rerank_score": chunk.get("rerank_score"),
                "rrf_score": chunk.get("rrf_score"),
            })
    return citations


def _stream_tokens(query: str, chunks: list[dict]) -> Iterator[str]:
    response = _get_client().chat.completions.create(
        messages=_build_messages(query, chunks), stream=True, **_llm_kwargs()
    )
    for event in response:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def generate_answer(query: str, chunks: list[dict], stream: bool = False) -> dict:
    if stream:
        # Lazy generator: the API call starts when the route begins iterating.
        return {"stream": _stream_tokens(query, chunks), "chunks": chunks}

    start = time.perf_counter()
    response = _get_client().chat.completions.create(
        messages=_build_messages(query, chunks), **_llm_kwargs()
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "citations": extract_citations(answer, chunks),
        "latency_ms": round(latency_ms, 2),
        "tokens_used": response.usage.total_tokens if response.usage else None,
        "chunks_retrieved": len(chunks),
    }
