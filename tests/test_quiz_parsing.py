"""
Quiz Parsing, Validation, and Deterministic Grading Tests (tests/test_quiz_parsing.py).
Tests:
- Stripping code fences and extracting JSON payloads
- Schema validation: exact 4 options, valid answer_index 0-3
- Graceful dropping of malformed questions
- Deterministic score computation in code without LLM hallucination
"""

import pytest
from app.quiz import (
    QuizItem,
    extract_json_block,
    grade_quiz,
    validate_quiz_json,
)


def test_extract_json_block_strips_markdown():
    """Verifies that markdown code blocks are cleanly stripped."""
    raw_markdown = "Here is your quiz:\n```json\n[{\"question\": \"Test?\"}]\n```\nHope you like it!"
    cleaned = extract_json_block(raw_markdown)
    assert cleaned == '[{"question": "Test?"}]'


def test_validate_quiz_json_valid_item():
    """Verifies that well-formed quiz JSON yields QuizItem instances."""
    json_text = """
    [
      {
        "question": "What is the rank parameter in LoRA?",
        "options": ["Matrix dimension bottleneck", "Learning rate multiplier", "Context length", "Layer count"],
        "answer_index": 0,
        "explanation": "Rank r is the inner dimension of the decomposed matrices.",
        "source": "Lab Manual 1"
      }
    ]
    """
    items = validate_quiz_json(json_text)
    assert len(items) == 1
    assert items[0].question == "What is the rank parameter in LoRA?"
    assert len(items[0].options) == 4
    assert items[0].answer_index == 0


def test_validate_quiz_json_drops_malformed():
    """Verifies that questions with fewer than 4 options or invalid answer_index are dropped."""
    json_text = """
    [
      {
        "question": "Invalid item with only 2 options",
        "options": ["A", "B"],
        "answer_index": 0,
        "explanation": "Too few options"
      },
      {
        "question": "Invalid item with out-of-bounds answer_index",
        "options": ["A", "B", "C", "D"],
        "answer_index": 5,
        "explanation": "Index is 5 which is > 3"
      },
      {
        "question": "Valid item",
        "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
        "answer_index": 2,
        "explanation": "Valid item"
      }
    ]
    """
    items = validate_quiz_json(json_text)
    assert len(items) == 1
    assert items[0].question == "Valid item"
    assert items[0].answer_index == 2


def test_validate_quiz_json_completely_invalid_syntax():
    """Verifies that syntax errors return an empty list without crashing."""
    bad_json = "```json\n[This is totally broken { json\n```"
    items = validate_quiz_json(bad_json)
    assert items == []


def test_deterministic_grading_logic():
    """Verifies 100% deterministic grading accuracy."""
    items = [
        QuizItem(
            question="Q1",
            options=["A", "B", "C", "D"],
            answer_index=1,
            explanation="Exp 1",
            source="Doc 1"
        ),
        QuizItem(
            question="Q2",
            options=["W", "X", "Y", "Z"],
            answer_index=3,
            explanation="Exp 2",
            source="Doc 2"
        ),
    ]

    # Student chooses correct for Q1 (idx 1) and incorrect for Q2 (idx 0 instead of 3)
    student_answers = [1, 0]
    score, total, results = grade_quiz(items, student_answers)

    assert total == 2
    assert score == 1
    assert results[0]["is_correct"] is True
    assert results[0]["selected_option"] == "B"
    assert results[1]["is_correct"] is False
    assert results[1]["selected_option"] == "W"
    assert results[1]["correct_option"] == "Z"
