"""
Quiz Generation, Validation, and Deterministic Grading (Experiment 6 & 8.3).
Handles multiple-choice questions (MCQs):
- Parsing and schema validation (stripping code fences, verifying 4 options, checking index bounds)
- Server-side quiz session storage (answer keys NEVER sent to client)
- 100% deterministic grading in code without LLM hallucination
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.config import settings
from app.logging_utils import logger


class QuizItem(BaseModel):
    """Schema for a fully validated multiple-choice question stored server-side."""
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    answer_index: int = Field(..., ge=0, le=3)
    explanation: str
    source: str


class QuizClientItem(BaseModel):
    """Client-facing MCQ schema. Notice answer_index is strictly omitted."""
    question_id: int
    question: str
    options: List[str]


class QuizSessionStore:
    """
    In-memory server-side store for active quizzes.
    Employs an LRU-like capacity cap (MAX_STORED_QUIZZES) to prevent unbounded memory growth.
    """

    def __init__(self, capacity: int = settings.MAX_STORED_QUIZZES):
        self.capacity = capacity
        self.store: Dict[str, Dict[str, Any]] = {}

    def save_quiz(self, items: List[QuizItem]) -> str:
        quiz_id = str(uuid.uuid4())
        # Evict oldest entry if capacity reached
        if len(self.store) >= self.capacity:
            oldest_key = next(iter(self.store))
            self.store.pop(oldest_key, None)

        self.store[quiz_id] = {
            "items": items,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        return quiz_id

    def get_quiz(self, quiz_id: str) -> Optional[List[QuizItem]]:
        entry = self.store.get(quiz_id)
        if not entry:
            return None
        return entry["items"]


# Global quiz store instance
quiz_store = QuizSessionStore()


def extract_json_block(text: str) -> str:
    """Extracts raw JSON payload from potentially markdown-fenced text."""
    cleaned = text.strip()
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def validate_quiz_json(raw_text: str) -> List[QuizItem]:
    """
    Parses LLM generation, validates fields, ensures 4 options and valid answer_index.
    Drops malformed items. Returns a list of validated QuizItem objects.
    """
    cleaned_json = extract_json_block(raw_text)
    try:
        data = json.loads(cleaned_json)
    except Exception as e:
        logger.warning(f"Failed to parse quiz JSON: {e}. Raw text preview: {raw_text[:100]}")
        return []

    if isinstance(data, dict):
        # In case the model wrapped items under a 'questions' key
        if "questions" in data and isinstance(data["questions"], list):
            data = data["questions"]
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    validated_items: List[QuizItem] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        q_text = str(item.get("question", "")).strip()
        raw_options = item.get("options")
        raw_idx = item.get("answer_index")
        explanation = str(item.get("explanation", "")).strip()
        source = str(item.get("source", "Course Document")).strip()

        if not q_text or not isinstance(raw_options, list) or len(raw_options) != 4:
            continue

        options = [str(opt).strip() for opt in raw_options]
        # Validate that all options are non-empty
        if any(len(opt) == 0 for opt in options):
            continue

        try:
            answer_idx = int(raw_idx)
            if not (0 <= answer_idx <= 3):
                continue
        except (TypeError, ValueError):
            continue

        validated_items.append(
            QuizItem(
                question=q_text,
                options=options,
                answer_index=answer_idx,
                explanation=explanation or "Derived from course materials.",
                source=source
            )
        )

    return validated_items


def grade_quiz(
    stored_items: List[QuizItem],
    student_answers: List[int]
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Performs 100% deterministic grading comparing selected indices against answer_index.
    Returns (score, total, detailed_results).
    """
    score = 0
    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(stored_items):
        student_choice = student_answers[idx] if idx < len(student_answers) else -1
        is_correct = (student_choice == item.answer_index)
        if is_correct:
            score += 1

        selected_text = item.options[student_choice] if 0 <= student_choice < 4 else "No answer / Invalid"
        correct_text = item.options[item.answer_index]

        results.append({
            "question_id": idx + 1,
            "question": item.question,
            "selected_index": student_choice,
            "selected_option": selected_text,
            "correct_index": item.answer_index,
            "correct_option": correct_text,
            "is_correct": is_correct,
            "explanation": item.explanation,
            "source": item.source
        })

    return score, len(stored_items), results
