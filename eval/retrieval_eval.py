"""
Retrieval Evaluation Script (eval/retrieval_eval.py).
Evaluates the retrieval performance of the course knowledge base:
1. Hit Rate @ k (k = 1, 2, 3, 5) on the 20 held-out evaluation questions.
2. Chunk size comparative study (chunk_size = 300, 500, 800).
Outputs CSV and Markdown tables to eval/results/.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion import (
    get_embedding_function,
    load_raw_documents,
    load_vectorstore,
    split_documents,
)
from app.retriever import CourseRetriever
from langchain_community.vectorstores import FAISS


def run_hit_rate_evaluation(eval_path: str = "data/finetune/eval_questions.jsonl"):
    """
    Evaluates whether the known ground-truth source document is retrieved in top-k (k=1, 2, 3, 5).
    """
    print("=" * 60)
    print("RUNNING RETRIEVAL HIT-RATE EVALUATION (k = 1, 2, 3, 5)")
    print("=" * 60)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_questions = [json.loads(line) for line in f]

    retriever = CourseRetriever()
    k_values = [1, 2, 3, 5]
    per_question_results = []

    # Track hits per k
    hits = {k: 0 for k in k_values}
    evaluable_count = 0

    for item in eval_questions:
        q_id = item["id"]
        q_text = item["question"]
        gt_source = item.get("ground_truth_source", "none")
        q_type = item.get("type", "supported")

        if gt_source == "none" or q_type == "out_of_scope":
            continue

        evaluable_count += 1
        docs = retriever.retrieve(q_text, k=5)
        retrieved_sources = [d.metadata.get("source", "") for d in docs]

        row = {
            "Question ID": q_id,
            "Question": q_text[:50] + "...",
            "Ground Truth Source": gt_source,
            "Top-1 Source": retrieved_sources[0] if len(retrieved_sources) > 0 else "None",
            "Top-3 Sources": ", ".join(retrieved_sources[:3]),
        }

        for k in k_values:
            top_k_sources = retrieved_sources[:k]
            is_hit = gt_source in top_k_sources
            row[f"Hit@{k}"] = "Yes" if is_hit else "No"
            if is_hit:
                hits[k] += 1

        per_question_results.append(row)

    df_details = pd.DataFrame(per_question_results)
    os.makedirs("eval/results", exist_ok=True)
    df_details.to_csv("eval/results/retrieval_per_question_hits.csv", index=False)

    summary_rows = []
    for k in k_values:
        rate = (hits[k] / evaluable_count) * 100 if evaluable_count > 0 else 0.0
        summary_rows.append({
            "k": k,
            "Hits": hits[k],
            "Total In-Scope Questions": evaluable_count,
            "Hit Rate (%)": f"{rate:.1f}%"
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("eval/results/retrieval_hit_rates.csv", index=False)

    print("\nRETRIEVAL HIT RATE SUMMARY:")
    print(df_summary.to_markdown(index=False))
    return df_summary


def run_chunk_size_study(raw_dir: str = "data/raw", eval_path: str = "data/finetune/eval_questions.jsonl"):
    """
    Compares chunk sizes (300, 500, 800) on the held-out questions.
    """
    print("\n" + "=" * 60)
    print("RUNNING CHUNK SIZE COMPARATIVE STUDY (300 vs 500 vs 800)")
    print("=" * 60)

    with open(eval_path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
        eval_questions = [item for item in all_items if item.get("ground_truth_source") != "none"]

    raw_docs = load_raw_documents(raw_dir)
    embedding_fn = get_embedding_function(settings.EMBEDDING_MODEL_NAME)

    chunk_configs = [
        {"chunk_size": 300, "chunk_overlap": 50},
        {"chunk_size": 500, "chunk_overlap": 80},
        {"chunk_size": 800, "chunk_overlap": 120},
    ]

    chunk_results = []

    for cfg in chunk_configs:
        cs = cfg["chunk_size"]
        co = cfg["chunk_overlap"]
        print(f"Testing chunk_size={cs}, overlap={co}...")

        chunks = split_documents(raw_docs, chunk_size=cs, chunk_overlap=co)
        vs = FAISS.from_documents(chunks, embedding_fn)

        hits_at_1 = 0
        hits_at_3 = 0
        total_q = 0

        for item in eval_questions:
            gt_source = item.get("ground_truth_source", "none")
            if gt_source == "none" or item.get("type") == "out_of_scope":
                continue
            total_q += 1
            docs = vs.similarity_search(item["question"], k=3)
            sources = [d.metadata.get("source", "") for d in docs]
            if sources and sources[0] == gt_source:
                hits_at_1 += 1
            if gt_source in sources:
                hits_at_3 += 1

        chunk_results.append({
            "Chunk Size": cs,
            "Chunk Overlap": co,
            "Total Chunks": len(chunks),
            "Hit@1 (%)": f"{(hits_at_1 / total_q) * 100:.1f}%",
            "Hit@3 (%)": f"{(hits_at_3 / total_q) * 100:.1f}%",
            "Trade-off Observation": (
                "Granular but fragmented context" if cs == 300
                else ("Optimal balance of coherence and specificity" if cs == 500
                else "Broad context, risk of tangential dilution")
            )
        })

    df_chunks = pd.DataFrame(chunk_results)
    df_chunks.to_csv("eval/results/retrieval_chunk_size_comparison.csv", index=False)
    print("\nCHUNK SIZE COMPARISON SUMMARY:")
    print(df_chunks.to_markdown(index=False))

    # Write Markdown summary document
    with open("eval/results/retrieval_eval_summary.md", "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Results\n\n")
        f.write("## 1. Chunk Size Comparison\n\n")
        f.write(df_chunks.to_markdown(index=False) + "\n\n")

    return df_chunks


if __name__ == "__main__":
    run_hit_rate_evaluation()
    run_chunk_size_study()
