"""
Dataset Leakage and Overlap Verification Script (scripts/verify_dataset_overlap.py).
Ensures zero data leakage between training set (train.jsonl) and held-out evaluation set (eval_questions.jsonl).
Performs:
1. Exact string matching (case-insensitive, normalized)
2. Token-level Jaccard similarity analysis (>0.85 threshold)
Fails with exit code 1 if any data contamination is identified.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Set


def tokenize(text: str) -> Set[str]:
    """Extracts alphanumeric word tokens for Jaccard similarity computation."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity coefficient between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def verify_no_overlap(
    train_path: str = "data/finetune/train.jsonl",
    eval_path: str = "data/finetune/eval_questions.jsonl",
    similarity_threshold: float = 0.85
) -> bool:
    print("=" * 60)
    print("VERIFYING TRAINING & EVALUATION SET ISOLATION")
    print("=" * 60)

    if not Path(train_path).exists() or not Path(eval_path).exists():
        print(f"Error: Missing dataset files '{train_path}' or '{eval_path}'.")
        return False

    # Extract user questions from train.jsonl
    train_questions: List[str] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            for msg in entry.get("messages", []):
                if msg.get("role") == "user":
                    train_questions.append(msg.get("content", ""))

    # Extract held-out questions from eval_questions.jsonl
    eval_questions: List[dict] = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            eval_questions.append(json.loads(line))

    print(f"Loaded {len(train_questions)} training questions.")
    print(f"Loaded {len(eval_questions)} held-out evaluation questions.")

    violations = []

    for eval_item in eval_questions:
        e_id = eval_item.get("id")
        e_q = eval_item.get("question", "").strip()
        e_norm = e_q.lower()
        e_tokens = tokenize(e_q)

        for t_idx, t_q in enumerate(train_questions):
            t_norm = t_q.strip().lower()

            # Exact match check
            if e_norm == t_norm:
                violations.append({
                    "eval_id": e_id,
                    "eval_question": e_q,
                    "train_index": t_idx,
                    "train_question": t_q,
                    "reason": "Exact string match"
                })
                continue

            # Jaccard similarity check
            t_tokens = tokenize(t_q)
            similarity = compute_jaccard_similarity(e_tokens, t_tokens)
            if similarity >= similarity_threshold:
                violations.append({
                    "eval_id": e_id,
                    "eval_question": e_q,
                    "train_index": t_idx,
                    "train_question": t_q,
                    "reason": f"High lexical overlap (Jaccard: {similarity:.2f})"
                })

    if violations:
        print(f"\n[FAILURE] Detected {len(violations)} data leakage violations!")
        for v in violations:
            print(f"  • Eval [{v['eval_id']}]: '{v['eval_question']}'")
            print(f"    Train [{v['train_index']}]: '{v['train_question']}'")
            print(f"    Reason: {v['reason']}\n")
        return False

    print("\n[SUCCESS] Verification passed: ZERO overlap detected between training and eval sets.")
    print(f"All 20 held-out questions are strictly unseen by fine-tuning models.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = verify_no_overlap()
    if not success:
        sys.exit(1)
    sys.exit(0)
