import re
import time
from app.config import settings

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context chunks.
- Cite every factual claim using [Source N] inline.
- If the answer is not in the context, say "I don't have enough context to answer this."
- Be concise but complete.
"""

def build_context_prompt(query: str, chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("filename") or chunk.get("metadata", {}).get("source", "Unknown")
        page = chunk.get("metadata", {}).get("page", "?")
        parts.append(f"[Source {i}] (from: {source}, page {page})\n{chunk['text']}")
    return f"Context:\n\n{'---'.join(parts)}\n\nQuestion: {query}"

def generate_answer(query: str, chunks: list[dict], stream: bool = False) -> dict:
    user_message = build_context_prompt(query, chunks)
    start = time.perf_counter()

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        max_tokens=settings.max_tokens,
        temperature=0.1
    )
    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens
    latency_ms = (time.perf_counter() - start) * 1000

    cited = set(int(m) for m in re.findall(r'\[Source (\d+)\]', answer))
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        if i in cited:
            citations.append({
                "source_n": i,
                "chunk_id": chunk.get("id"),
                "faiss_id": chunk.get("faiss_id"),
                "filename": chunk.get("filename") or chunk.get("metadata", {}).get("source", "Unknown"),
                "page": chunk.get("metadata", {}).get("page"),
                "text_preview": chunk["text"][:200],
                "rerank_score": chunk.get("rerank_score"),
                "rrf_score": chunk.get("rrf_score"),
            })

    return {
        "answer": answer,
        "citations": citations,
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens,
        "chunks_retrieved": len(chunks),
    }
