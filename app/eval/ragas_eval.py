"""
RAGAS evaluation pipeline.
Measures: faithfulness, answer_relevancy, context_recall, context_precision.
Logs results to MLflow.

Usage:
  python -m app.eval.ragas_eval --qa-file app/eval/qa_pairs.json
"""
import json
import argparse
import time
from pathlib import Path

import mlflow
from datasets import Dataset

from app.config import settings
from app.core.retriever import get_retriever
from app.core.generator import generate_answer
from app.database import init_db


def load_qa_pairs(filepath: str) -> list[dict]:
    """Load Q&A pairs from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def run_rag_pipeline(query: str, retriever, top_k_rerank: int = 5) -> tuple[str, list[str]]:
    """
    Run retrieval + generation for a single query.
    Returns (answer, list_of_context_strings).
    """
    chunks = retriever.retrieve(query=query, top_k_rerank=top_k_rerank)
    if not chunks:
        return "No context available.", []

    result = generate_answer(query, chunks, stream=False)
    contexts = [c["text"] for c in chunks]
    return result["answer"], contexts


def evaluate(qa_pairs_path: str, top_k_rerank: int = 5, experiment_name: str = None):
    """
    Run RAGAS evaluation over all Q&A pairs.
    Logs results to MLflow.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
        return

    init_db()
    retriever = get_retriever()
    qa_pairs = load_qa_pairs(qa_pairs_path)
    print(f"[RAGAS] Evaluating {len(qa_pairs)} Q&A pairs...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, pair in enumerate(qa_pairs):
        print(f"  [{i+1}/{len(qa_pairs)}] {pair['question'][:60]}...")
        answer, ctx = run_rag_pipeline(pair["question"], retriever, top_k_rerank)
        questions.append(pair["question"])
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(pair["ground_truth"])
        time.sleep(0.5)  # rate limit

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = ragas_evaluate(dataset, metrics=metrics)

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }

    print("\n=== RAGAS Results ===")
    for metric, score in scores.items():
        print(f"  {metric:25s}: {score:.4f}")

    # Log to MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    exp_name = experiment_name or settings.mlflow_experiment_name
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=f"ragas_eval_{int(time.time())}"):
        mlflow.log_params({
            "top_k_rerank": top_k_rerank,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "num_qa_pairs": len(qa_pairs),
        })
        mlflow.log_metrics(scores)

        # Save detailed results
        result_path = Path("./data/ragas_results.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump({
                "scores": scores,
                "qa_pairs_evaluated": len(qa_pairs),
                "config": {
                    "embedding_model": settings.embedding_model,
                    "llm_model": settings.llm_model,
                    "top_k_rerank": top_k_rerank,
                }
            }, f, indent=2)

        mlflow.log_artifact(str(result_path))

    print(f"\n[MLflow] Results logged to: {settings.mlflow_tracking_uri}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--qa-file", default="app/eval/qa_pairs.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    evaluate(args.qa_file, args.top_k, args.experiment)
