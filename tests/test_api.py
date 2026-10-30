"""
Integration Tests for StudyBuddy FastAPI Endpoints (tests/test_api.py).
Uses TestClient with mock LLM backend. Runs on CPU with no network requirements.
"""

import os
# Force mock backend for reproducible, offline unit testing
os.environ["LLM_BACKEND"] = "mock"

import pytest
from fastapi.testclient import TestClient

from app.config import settings
settings.LLM_BACKEND = "mock"

from app.main import app
from app.prompts import FALLBACK_TEXT


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient with lifespan events executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Test that /health returns success, index loaded, and active backend."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True
    assert data["backend"] == "mock"


def test_ask_valid_concept_question(client: TestClient):
    """Test valid concept question returns answer with source citation and latency."""
    payload = {"question": "What is LoRA and how does rank r affect parameter count?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data and len(data["answer"]) > 10
    assert data["intent"] == "explain_concept"
    assert "sources" in data
    assert len(data["sources"]) >= 1
    assert "source" in data["sources"][0]
    assert "page" in data["sources"][0]
    assert data["latency_ms"] > 0
    assert "X-Response-Time-ms" in response.headers


def test_ask_unsupported_question_triggers_fallback(client: TestClient):
    """Test that out-of-scope question triggers fallback text instead of invented facts."""
    payload = {"question": "What is the live weather forecast in Tokyo today?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["intent"] == "out_of_scope"
    assert FALLBACK_TEXT in data["answer"]


def test_quiz_generation_and_grading_flow(client: TestClient):
    """Test full quiz lifecycle: generate quiz (no answers), submit, deterministic grading."""
    # Step 1: Generate Quiz
    gen_payload = {"topic": "LoRA and QLoRA Quantization", "num_questions": 2}
    gen_res = client.post("/quiz", json=gen_payload)
    assert gen_res.status_code == 200

    quiz_data = gen_res.json()
    quiz_id = quiz_data["quiz_id"]
    questions = quiz_data["questions"]

    assert len(quiz_id) > 10
    assert len(questions) == 2
    for q in questions:
        assert "question_id" in q
        assert "question" in q
        assert "options" in q
        assert len(q["options"]) == 4
        # Crucial security & grading check: Answer keys must NOT be returned to client!
        assert "answer_index" not in q
        assert "explanation" not in q

    # Step 2: Grade Quiz
    # In mock backend, option index 1 is the intended answer for the generated items
    grade_payload = {"quiz_id": quiz_id, "answers": [1, 1]}
    grade_res = client.post("/quiz/grade", json=grade_payload)
    assert grade_res.status_code == 200

    grade_data = grade_res.json()
    assert grade_data["quiz_id"] == quiz_id
    assert grade_data["score"] == 2
    assert grade_data["total"] == 2
    assert grade_data["percentage"] == 100.0
    assert len(grade_data["results"]) == 2
    assert grade_data["results"][0]["is_correct"] is True
    assert "explanation" in grade_data["results"][0]
    assert "source" in grade_data["results"][0]


def test_grade_quiz_unknown_id_returns_404(client: TestClient):
    """Test grading with non-existent quiz_id returns HTTP 404."""
    payload = {"quiz_id": "non-existent-quiz-uuid-12345", "answers": [0, 1]}
    response = client.post("/quiz/grade", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_interview_question_generation(client: TestClient):
    """Test requesting an interview question without student answer."""
    payload = {"topic": "Parameter-Efficient Fine-Tuning"}
    response = client.post("/interview", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "feedback_or_question" in data
    assert len(data["feedback_or_question"]) > 10
    assert data["latency_ms"] > 0


def test_interview_evaluation_rubric(client: TestClient):
    """Test evaluating student answer against rubric."""
    payload = {
        "question": "Explain how LoRA reduces parameter memory requirements.",
        "student_answer": "LoRA freezes the base model and trains low-rank decomposition matrices B and A."
    }
    response = client.post("/interview", json=payload)
    assert response.status_code == 200

    data = response.json()
    feedback = data["feedback_or_question"]
    assert "VERDICT:" in feedback
    assert "WHAT WAS GOOD:" in feedback
    assert "WHAT WAS MISSING:" in feedback
    assert "MODEL ANSWER:" in feedback


def test_backend_failure_sanitized_error(client: TestClient):
    """Test simulated backend failure returns generic HTTP 500 without leaking stack traces or keys."""
    payload = {"question": "What is LoRA? SIMULATE_FAILURE"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 500

    data = response.json()
    assert "detail" in data
    # Guarantee no traceback or internal code details leaked
    assert "Traceback" not in data["detail"]
    assert "Exception" not in data["detail"]
    assert "/" not in data["detail"]
    assert "\\" not in data["detail"]
