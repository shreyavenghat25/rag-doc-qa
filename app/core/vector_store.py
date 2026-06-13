"""
FAISS index using HNSWFlat for approximate nearest-neighbour search.
HNSWFlat: no training needed, good recall, fast at query time.
"""
import os
import numpy as np
import faiss
from pathlib import Path
from app.config import settings


class FAISSIndex:
    def __init__(self, dimension: int, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or settings.faiss_index_path
        self.index = self._load_or_create()

    def _load_or_create(self) -> faiss.Index:
        path = Path(self.index_path)
        if path.exists():
            print(f"[FAISS] Loading existing index from {path}")
            return faiss.read_index(str(path))
        else:
            print(f"[FAISS] Creating new HNSWFlat index (dim={self.dimension})")
            # M=16: number of neighbours in HNSW graph
            # efConstruction=200: accuracy/speed tradeoff during build
            index = faiss.IndexHNSWFlat(self.dimension, 16)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64  # query-time accuracy
            return index

    def add(self, vectors: np.ndarray) -> list[int]:
        """
        Add vectors to index.
        Returns list of FAISS ids assigned (sequential from current ntotal).
        """
        start_id = self.index.ntotal
        vectors = vectors.astype(np.float32)
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)
        self.index.add(vectors)
        end_id = self.index.ntotal
        return list(range(start_id, end_id))

    def search(self, query_vector: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """
        Search index for k nearest neighbours.
        Returns (faiss_ids, distances).
        Distances are L2 for HNSWFlat with normalized vectors ≈ 2*(1-cosine_sim).
        """
        query = query_vector.astype(np.float32).reshape(1, -1)
        if not query.flags["C_CONTIGUOUS"]:
            query = np.ascontiguousarray(query)
        distances, ids = self.index.search(query, k)
        ids = ids[0].tolist()
        distances = distances[0].tolist()
        # Filter out -1 (FAISS returns -1 when fewer than k results)
        valid = [(i, d) for i, d in zip(ids, distances) if i != -1]
        if not valid:
            return [], []
        ids, distances = zip(*valid)
        return list(ids), list(distances)

    def save(self):
        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"[FAISS] Saved index to {path} ({self.index.ntotal} vectors)")

    @property
    def size(self) -> int:
        return self.index.ntotal


# Singleton
_faiss_index: FAISSIndex | None = None


def get_faiss_index(dimension: int = None) -> FAISSIndex:
    global _faiss_index
    if _faiss_index is None:
        from app.core.embedder import get_embedder
        dim = dimension or get_embedder().dimension
        _faiss_index = FAISSIndex(dimension=dim)
    return _faiss_index
