"""
Input Validation and Boundary Tests (tests/test_validation.py).
Verifies Pydantic schema constraints:
- Rejection of empty or whitespace-only inputs
- Rejection of inputs exceeding max character length (1000 chars)
- Bounds on quiz question count (1 to 5)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_reject_empty_question(client: TestClient):
    """Test empty string question is rejected with 422."""
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_reject_whitespace_only_question(client: TestClient):
    """Test whitespace-only question is rejected with 422."""
    res = client.post("/ask", json={"question": "   \n\t   "})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_reject_overlength_question(client: TestClient):
    """Test question longer than 1000 characters is rejected with 422."""
    long_q = "What is LoRA? " * 100  # 1400 characters
    assert len(long_q) > 1000
    res = client.post("/ask", json={"question": long_q})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_quiz_num_questions_bounds(client: TestClient):
    """Test num_questions below 1 or above 5 are rejected with 422."""
    # Test 0
    res_zero = client.post("/quiz", json={"topic": "LoRA", "num_questions": 0})
    assert res_zero.status_code == 422

    # Test 6
    res_six = client.post("/quiz", json={"topic": "LoRA", "num_questions": 6})
    assert res_six.status_code == 422

    # Test valid boundary values 1 and 5
    res_one = client.post("/quiz", json={"topic": "LoRA", "num_questions": 1})
    assert res_one.status_code == 200

    res_five = client.post("/quiz", json={"topic": "LoRA", "num_questions": 5})
    assert res_five.status_code == 200


def test_quiz_empty_topic(client: TestClient):
    """Test empty quiz topic is rejected with 422."""
    res = client.post("/quiz", json={"topic": "   ", "num_questions": 3})
    assert res.status_code == 422
