"""
Two chunking strategies:
1. RecursiveCharacterTextSplitter (baseline)
2. Semantic chunking - splits where cosine similarity drops between sentences
"""
import re
import numpy as np
from app.config import settings


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple rules."""
    # Split on . ! ? followed by whitespace or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def recursive_character_split(
    pages: list[dict],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> list[dict]:
    """
    Baseline chunker: split by character count with overlap.
    Respects paragraph boundaries when possible.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks = []
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        result = []
        current = ""
        for part in splits:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    result.extend(_split(part, separators[1:]))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result

    for page in pages:
        text = page["text"]
        raw_chunks = _split(text, separators)

        # Apply overlap
        overlapped = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and chunk_overlap > 0:
                prev = raw_chunks[i - 1]
                overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                chunk = overlap_text + " " + chunk
            overlapped.append(chunk)

        for i, chunk_text in enumerate(overlapped):
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {**page["metadata"], "chunk_strategy": "recursive"}
            })

    return chunks


def semantic_chunk(
    pages: list[dict],
    embedder,
    breakpoint_percentile: float = 85.0
) -> list[dict]:
    """
    Semantic chunking: embed sentences, find natural split points
    where cosine similarity between adjacent sentences drops sharply.
    Falls back to recursive chunking for very short documents.
    """
    all_sentences = []
    sentence_meta = []

    for page in pages:
        sentences = _split_sentences(page["text"])
        for s in sentences:
            if len(s) > 20:  # skip trivially short sentences
                all_sentences.append(s)
                sentence_meta.append(page["metadata"])

    if len(all_sentences) < 4:
        return recursive_character_split(pages)

    # Embed all sentences
    embeddings = embedder.embed_batch(all_sentences)

    # Compute cosine similarity between adjacent sentences
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    similarities = [
        cosine_sim(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Breakpoints where similarity drops below percentile threshold
    threshold = np.percentile(similarities, 100 - breakpoint_percentile)
    split_indices = [i + 1 for i, s in enumerate(similarities) if s < threshold]

    # Build chunks from sentence groups
    chunks = []
    boundaries = [0] + split_indices + [len(all_sentences)]

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        group = all_sentences[start:end]
        chunk_text = " ".join(group).strip()

        if len(chunk_text) > settings.chunk_size * 2:
            # Chunk is too long — split recursively
            sub = recursive_character_split(
                [{"text": chunk_text, "metadata": sentence_meta[start]}]
            )
            chunks.extend(sub)
        else:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **sentence_meta[start],
                    "chunk_strategy": "semantic",
                    "sentence_count": end - start
                }
            })

    return chunks
