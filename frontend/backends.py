"""
Two interchangeable backends for the Streamlit UI.

ApiBackend      — talks to the FastAPI service over HTTP (local dev, Docker).
EmbeddedBackend — runs the RAG pipeline in-process with a private per-visitor
                  index (hosted demo on Streamlit Community Cloud, which runs
                  a single Streamlit process and no separate API server).

Both expose the same methods and raise BackendError with a readable message,
so the UI code doesn't care which one it is using.
"""
import json
from typing import Iterator

import requests

REQUEST_TIMEOUT = 120  # the first query loads the reranker model


class BackendError(Exception):
    pass


class ApiBackend:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _post(self, path: str, **kwargs) -> dict:
        try:
            r = requests.post(f"{self.base}/{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as e:
            raise BackendError(f"API not reachable at {self.base}: {e}") from e
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise BackendError(f"Request failed ({r.status_code}): {detail}")
        return r.json()

    def index_pdf(self, data: bytes, name: str, semantic: bool) -> dict:
        return self._post("upload", files={"file": (name, data, "application/pdf")},
                          params={"use_semantic_chunking": semantic})

    def index_url(self, url: str, semantic: bool) -> dict:
        return self._post("index/url", json={"url": url, "use_semantic_chunking": semantic})

    def index_text(self, text: str, name: str) -> dict:
        return self._post("index/text", json={"text": text, "filename": name})

    def list_documents(self) -> list[dict]:
        try:
            r = requests.get(f"{self.base}/documents", timeout=3)
            r.raise_for_status()
        except requests.RequestException as e:
            raise BackendError(f"API not reachable at {self.base}") from e
        return r.json()["documents"]

    def query(self, query: str, top_k_retrieve: int, top_k_rerank: int, use_reranker: bool) -> dict:
        return self._post("query", json={
            "query": query, "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank, "use_reranker": use_reranker,
        })

    def query_stream(self, query: str, top_k_retrieve: int, top_k_rerank: int) -> Iterator[dict]:
        params = {"query": query, "top_k_retrieve": top_k_retrieve, "top_k_rerank": top_k_rerank}
        try:
            with requests.get(f"{self.base}/query/stream", params=params,
                              stream=True, timeout=REQUEST_TIMEOUT) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
        except requests.RequestException as e:
            raise BackendError(f"Streaming request failed: {e}") from e


class EmbeddedBackend:
    def __init__(self, session):
        self.session = session  # app.core.session.RAGSession

    @staticmethod
    def _wrap(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            raise BackendError(str(e)) from e
        except Exception as e:  # network errors on URL fetch, PDF parse errors, etc.
            raise BackendError(f"{type(e).__name__}: {e}") from e

    def index_pdf(self, data: bytes, name: str, semantic: bool) -> dict:
        return self._wrap(self.session.index_pdf, data, name, semantic)

    def index_url(self, url: str, semantic: bool) -> dict:
        return self._wrap(self.session.index_url, url, semantic)

    def index_text(self, text: str, name: str) -> dict:
        return self._wrap(self.session.index_text, text, name)

    def list_documents(self) -> list[dict]:
        return self.session.list_documents()

    def query(self, query: str, top_k_retrieve: int, top_k_rerank: int, use_reranker: bool) -> dict:
        return self._wrap(self.session.query, query, top_k_retrieve, top_k_rerank, use_reranker)

    def query_stream(self, query: str, top_k_retrieve: int, top_k_rerank: int) -> Iterator[dict]:
        try:
            yield from self.session.query_stream(query, top_k_retrieve, top_k_rerank)
        except Exception as e:
            raise BackendError(f"{type(e).__name__}: {e}") from e
