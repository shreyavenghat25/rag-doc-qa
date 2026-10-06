"""
Retrieval ablation benchmark on a BEIR dataset (default: SciFact).

Compares the four retrieval configurations used by the app, on the same index:
  bm25            lexical only
  dense           FAISS + bge-small only
  hybrid          BM25 + dense fused with Reciprocal Rank Fusion
  hybrid+rerank   hybrid candidates re-scored by the cross-encoder (the app's default)

Metrics (standard BEIR/TREC definitions): nDCG@10, MRR@10, Recall@5/10/20,
plus mean per-query latency on this machine.

Usage:
  python -m app.eval.retrieval_eval                    # full SciFact test set (300 queries)
  python -m app.eval.retrieval_eval --max-queries 50   # quick run
  python -m app.eval.retrieval_eval --dataset nfcorpus

Writes app/eval/results/retrieval_<dataset>.{json,md}.
"""
import argparse
import csv
import io
import json
import math
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATA_ROOT = Path("data/beir")
RESULTS_DIR = Path(__file__).parent / "results"
CANDIDATES = 20  # same candidate depth the app uses (settings.top_k_retrieve)


# ── Data ─────────────────────────────────────────────────────────────────────
def _download_from_hf(name: str, target: Path) -> None:
    """BEIR mirrors on Hugging Face: BeIR/<name> (corpus, queries) and BeIR/<name>-qrels."""
    import gzip
    import shutil
    from huggingface_hub import hf_hub_download

    (target / "qrels").mkdir(parents=True, exist_ok=True)
    for fname in ("corpus.jsonl.gz", "queries.jsonl.gz"):
        src = hf_hub_download(f"BeIR/{name}", fname, repo_type="dataset")
        with gzip.open(src, "rb") as fin, open(target / fname[:-3], "wb") as fout:
            shutil.copyfileobj(fin, fout)
    src = hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset")
    shutil.copy(src, target / "qrels" / "test.tsv")


def _download_from_ukp(name: str) -> None:
    url = BEIR_URL.format(name=name)
    with urllib.request.urlopen(url, timeout=120) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA_ROOT)


def download_beir(name: str) -> Path:
    target = DATA_ROOT / name
    if (target / "qrels" / "test.tsv").exists():
        return target
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[data] Downloading BeIR/{name} from Hugging Face")
        _download_from_hf(name, target)
    except Exception as e:
        print(f"[data] Hugging Face download failed ({e}); trying the BEIR server")
        _download_from_ukp(name)
    return target


def load_beir(path: Path, split: str = "test"):
    corpus = {}
    with open(path / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + "\n" + d.get("text", "")).strip()
    queries = {}
    with open(path / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return corpus, queries, qrels


# ── Metrics ──────────────────────────────────────────────────────────────────
def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & rels.keys()) / len(rels)


def score_run(runs: dict[str, list[str]], qrels: dict) -> dict:
    n = len(runs)
    return {
        "nDCG@10": sum(ndcg_at_k(r, qrels[q]) for q, r in runs.items()) / n,
        "MRR@10": sum(mrr_at_k(r, qrels[q]) for q, r in runs.items()) / n,
        "Recall@5": sum(recall_at_k(r, qrels[q], 5) for q, r in runs.items()) / n,
        "Recall@10": sum(recall_at_k(r, qrels[q], 10) for q, r in runs.items()) / n,
        "Recall@20": sum(recall_at_k(r, qrels[q], 20) for q, r in runs.items()) / n,
    }


# ── Index ────────────────────────────────────────────────────────────────────
def build_index(corpus: dict[str, str]):
    """Index every document as one chunk, using the app's own components."""
    from app.core.embedder import get_embedder
    from app.core.retriever import HybridRetriever
    from app.core.session import InMemoryStore
    from app.core.vector_store import FAISSIndex

    embedder = get_embedder()
    index = FAISSIndex(dimension=embedder.dimension, persist=False)
    store = InMemoryStore()
    doc_ids = list(corpus)
    texts = [corpus[d] for d in doc_ids]

    t0 = time.perf_counter()
    cache = DATA_ROOT / f"emb_{embedder.model_name.replace('/', '_')}_{len(texts)}.npy"
    if cache.exists():
        print(f"[index] Loading cached embeddings from {cache}")
        vectors = np.load(cache)
    else:
        print(f"[index] Embedding {len(texts)} documents…")
        vectors = embedder.embed_batch(texts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)  # reruns skip the slow step
    faiss_ids = index.add(vectors)
    chunks = [{"text": t, "metadata": {"doc_id": d}} for t, d in zip(texts, doc_ids)]
    store.add_document("beir", "benchmark", chunks, faiss_ids)
    retriever = HybridRetriever(embedder, index, store=store)
    retriever.refresh_bm25()
    print(f"[index] Done in {time.perf_counter() - t0:.0f}s")

    faiss_to_doc = dict(zip(faiss_ids, doc_ids))
    return retriever, faiss_to_doc


# ── Ablation ─────────────────────────────────────────────────────────────────
def run_configs(retriever, faiss_to_doc, queries: dict[str, str]) -> tuple[dict, dict]:
    def ids_from_pairs(pairs):
        return [faiss_to_doc[f] for f, _ in pairs]

    def ids_from_chunks(chunks):
        return [c["metadata"]["doc_id"] for c in chunks]

    configs = {
        "bm25": lambda q: ids_from_pairs(retriever._sparse_search(q, CANDIDATES)),
        "dense": lambda q: ids_from_pairs(retriever._dense_search(q, CANDIDATES)),
        "hybrid": lambda q: ids_from_chunks(retriever.retrieve(
            q, top_k_retrieve=CANDIDATES, top_k_rerank=CANDIDATES, use_reranker=False)),
        "hybrid+rerank": lambda q: ids_from_chunks(retriever.retrieve(
            q, top_k_retrieve=CANDIDATES, top_k_rerank=CANDIDATES, use_reranker=True)),
    }

    # Warm-up so model loading isn't counted as query latency.
    first = next(iter(queries.values()))
    for fn in configs.values():
        fn(first)

    runs, latency = {}, {}
    for name, fn in configs.items():
        print(f"[eval] {name} on {len(queries)} queries…")
        t0 = time.perf_counter()
        runs[name] = {qid: fn(q) for qid, q in queries.items()}
        latency[name] = (time.perf_counter() - t0) * 1000 / len(queries)
    return runs, latency


def to_markdown(dataset: str, n_docs: int, n_queries: int, scores: dict, latency: dict) -> str:
    cols = ["nDCG@10", "MRR@10", "Recall@5", "Recall@10", "Recall@20"]
    lines = [
        f"**BEIR {dataset}** · {n_docs:,} documents · {n_queries} test queries · "
        f"{CANDIDATES} candidates per query",
        "",
        "| Configuration | " + " | ".join(cols) + " | Latency / query |",
        "|" + "---|" * (len(cols) + 2),
    ]
    for name, s in scores.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{s[c]:.3f}" for c in cols) + f" | {latency[name]:.0f} ms |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Retrieval ablation on a BEIR dataset")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", help="Use an already-downloaded BEIR folder")
    parser.add_argument("--max-queries", type=int, help="Evaluate a random subset (fixed seed)")
    args = parser.parse_args()

    path = Path(args.data_dir) if args.data_dir else download_beir(args.dataset)
    corpus, queries, qrels = load_beir(path)
    if args.max_queries and args.max_queries < len(queries):
        keep = random.Random(42).sample(sorted(queries), args.max_queries)
        queries = {q: queries[q] for q in keep}
    print(f"[data] {len(corpus):,} documents, {len(queries)} queries")

    retriever, faiss_to_doc = build_index(corpus)
    runs, latency = run_configs(retriever, faiss_to_doc, queries)
    scores = {name: score_run(run, qrels) for name, run in runs.items()}

    table = to_markdown(args.dataset, len(corpus), len(queries), scores, latency)
    print("\n" + table + "\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"retrieval_{args.dataset}"
    stem.with_suffix(".md").write_text(table + "\n")
    stem.with_suffix(".json").write_text(json.dumps({
        "dataset": args.dataset, "documents": len(corpus), "queries": len(queries),
        "candidates": CANDIDATES, "scores": scores, "latency_ms": latency,
    }, indent=2))
    print(f"[done] Saved {stem}.md and {stem}.json")

    try:  # optional experiment tracking
        import mlflow
        from app.config import settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(f"{settings.mlflow_experiment_name}-retrieval")
        for name, s in scores.items():
            with mlflow.start_run(run_name=f"{args.dataset}-{name}"):
                mlflow.log_params({"dataset": args.dataset, "config": name,
                                   "queries": len(queries), "candidates": CANDIDATES,
                                   "embedding_model": settings.embedding_model})
                mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in s.items()}
                                   | {"latency_ms": latency[name]})
        print("[mlflow] Logged runs (view with: mlflow ui)")
    except Exception as e:
        print(f"[mlflow] Skipped: {e}")


if __name__ == "__main__":
    main()
