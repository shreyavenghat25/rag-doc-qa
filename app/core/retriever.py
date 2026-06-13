"""
Hybrid retrieval pipeline:
1. BM25 sparse retrieval  (lexical match)
2. FAISS dense retrieval  (semantic match)
3. Reciprocal Rank Fusion (combine rankings)
4. Cross-encoder reranking (precise scoring on top-k)
"""
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.config import settings
from app.database import get_chunks_by_faiss_ids, get_all_chunks


class HybridRetriever:
    def __init__(self, embedder, faiss_index):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] | None = None  # all chunks for BM25
        self._reranker: CrossEncoder | None = None

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print("[Retriever] Loading cross-encoder reranker...")
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def _build_bm25(self):
        """Build or rebuild BM25 index from all chunks in DB."""
        self._corpus = get_all_chunks()
        if not self._corpus:
            self._bm25 = None
            return
        tokenized = [chunk["text"].lower().split() for chunk in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def refresh_bm25(self):
        """Call after adding new documents."""
        self._build_bm25()

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns list of (faiss_id, score). Score = 1 - L2/2 (higher = better)."""
        query_vec = self.embedder.embed(query)
        ids, distances = self.faiss_index.search(query_vec, k)
        # Convert L2 distance to similarity score (normalized vectors: L2² = 2-2*cos)
        results = []
        for faiss_id, dist in zip(ids, distances):
            score = 1.0 - dist / 2.0  # cosine similarity approximation
            results.append((faiss_id, score))
        return results

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """BM25 search. Returns list of (faiss_id, bm25_score)."""
        if self._bm25 is None:
            self._build_bm25()
        if self._bm25 is None or not self._corpus:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._corpus[idx]["faiss_id"], float(scores[idx])))
        return results

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[int, float]]],
        k: int = 60
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank(d))
        k=60 is standard default from the RRF paper.
        Returns sorted list of (faiss_id, rrf_score).
        """
        rrf_scores: dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, (faiss_id, _) in enumerate(ranked_list, start=1):
                rrf_scores[faiss_id] = rrf_scores.get(faiss_id, 0.0) + 1.0 / (k + rank)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Cross-encoder reranking: scores each (query, chunk) pair precisely.
        Much slower than embedding similarity but far more accurate.
        """
        reranker = self._get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k_retrieve: int = None,
        top_k_rerank: int = None,
        rrf_k: int = None,
        use_reranker: bool = True
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.
        Returns top_k_rerank chunks with scores.
        """
        top_k_retrieve = top_k_retrieve or settings.top_k_retrieve
        top_k_rerank = top_k_rerank or settings.top_k_rerank
        rrf_k = rrf_k or settings.rrf_k

        # Step 1: Dense retrieval
        dense_results = self._dense_search(query, top_k_retrieve)

        # Step 2: Sparse retrieval
        sparse_results = self._sparse_search(query, top_k_retrieve)

        if not dense_results and not sparse_results:
            return []

        # Step 3: RRF fusion
        fused = self._reciprocal_rank_fusion(
            [r for r in [dense_results, sparse_results] if r],
            k=rrf_k
        )

        # Get top candidates for reranking
        candidate_faiss_ids = [faiss_id for faiss_id, _ in fused[:top_k_retrieve]]
        candidates = get_chunks_by_faiss_ids(candidate_faiss_ids)

        # Add RRF scores to candidates
        rrf_score_map = {faiss_id: score for faiss_id, score in fused}
        for c in candidates:
            c["rrf_score"] = rrf_score_map.get(c["faiss_id"], 0.0)

        if not candidates:
            return []

        # Step 4: Cross-encoder reranking
        if use_reranker and len(candidates) > 1:
            return self._rerank(query, candidates, top_k_rerank)

        # Fallback: sort by RRF score
        return sorted(candidates, key=lambda x: x["rrf_score"], reverse=True)[:top_k_rerank]


# Singleton
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        from app.core.embedder import get_embedder
        from app.core.vector_store import get_faiss_index
        _retriever = HybridRetriever(get_embedder(), get_faiss_index())
    return _retriever
