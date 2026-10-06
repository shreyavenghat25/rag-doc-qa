from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "low"  # only used by gpt-oss models
    max_tokens: int = 1024
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    faiss_index_path: str = "./data/faiss_index"
    sqlite_db_path: str = "./data/metadata.db"
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "rag-doc-qa"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
Path("./data").mkdir(parents=True, exist_ok=True)
