"""
Embedding wrapper using sentence-transformers.
Default: BAAI/bge-small-en-v1.5 (free, strong performance).
"""
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import settings


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        print(f"[Embedder] Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready. Dimension: {self.dimension}")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns normalized float32 array."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of strings. Returns (N, dim) float32 array."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return vecs.astype(np.float32)


# Singleton — load once, reuse everywhere
@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
