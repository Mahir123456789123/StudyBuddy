"""
Public Facade for StudyBuddy RAG and Graph Operations (used by FastAPI in app/main.py).
Provides clean, reusable functions for:
- Asking concept questions (/ask)
- Generating quizzes (/quiz)
- Grading quiz responses (/quiz/grade)
- Practicing interview questions and evaluations (/interview)
"""

from typing import Any, Dict, List, Optional
from app.graph import StudyBuddyWorkflow, TutorState
from app.quiz import QuizClientItem, QuizItem, grade_quiz, quiz_store


def ask_concept(
    question: str,
    workflow: StudyBuddyWorkflow
) -> Dict[str, Any]:
    """
    Executes the LangGraph pipeline for a student's concept query.
    Returns grounded answer, classified intent, citations, and retry count.
    """
    initial_state: TutorState = {
        "question": question,
        "mode": "ask",
        "retry_count": 0
    }
    result = workflow.run(initial_state)

    return {
        "answer": result.get("final_answer") or result.get("draft") or "",
        "intent": result.get("intent", "explain_concept"),
        "sources": result.get("sources", []),
        "retries": result.get("retry_count", 0)
    }


def generate_quiz_session(
    topic: str,
    num_questions: int,
    workflow: StudyBuddyWorkflow
) -> Dict[str, Any]:
    """
    Generates MCQ questions, stores answer keys server-side, and returns
    sanitized questions without answer keys to the client.
    """
    initial_state: TutorState = {
        "topic": topic,
        "question": f"Generate a quiz on {topic}",
        "num_questions": num_questions,
        "mode": "quiz",
        "retry_count": 0
    }
    result = workflow.run(initial_state)
    raw_items = result.get("quiz", [])

    quiz_items: List[QuizItem] = [
        QuizItem(**item) if isinstance(item, dict) else item
        for item in raw_items
    ]

    if not quiz_items:
        # Fallback items if LLM generation was completely empty
        quiz_items = [
            QuizItem(
                question=f"What is a primary principle of {topic}?",
                options=[
                    f"Parameter efficiency in {topic}",
                    "Ignoring all loss functions",
                    "Replacing neural networks with regex",
                    "Zero memory usage"
                ],
                answer_index=0,
                explanation=f"A key objective of {topic} is computational and parameter efficiency.",
                source="Course Materials"
            )
        ]

    # Save to server-side memory store
    quiz_id = quiz_store.save_quiz(quiz_items)

    # Sanitize for client presentation: strictly exclude answer_index, explanation, source
    client_questions: List[Dict[str, Any]] = [
        {
            "question_id": idx + 1,
            "question": item.question,
            "options": item.options
        }
        for idx, item in enumerate(quiz_items)
    ]

    return {
        "quiz_id": quiz_id,
        "questions": client_questions
    }


def grade_quiz_session(
    quiz_id: str,
    student_answers: List[int]
) -> Dict[str, Any]:
    """
    Deterministically grades student responses against server-stored answer keys.
    Raises KeyError if quiz_id is unknown or expired.
    """
    stored_items = quiz_store.get_quiz(quiz_id)
    if stored_items is None:
        raise KeyError(f"Quiz ID '{quiz_id}' not found or session has expired.")

    score, total, detailed_results = grade_quiz(stored_items, student_answers)

    return {
        "quiz_id": quiz_id,
        "score": score,
        "total": total,
        "percentage": round((score / total) * 100, 1) if total > 0 else 0.0,
        "results": detailed_results
    }


def practice_interview(
    topic: Optional[str],
    question: Optional[str],
    student_answer: Optional[str],
    workflow: StudyBuddyWorkflow
) -> Dict[str, Any]:
    """
    Executes technical interview practice:
    - If student_answer is provided: evaluates response against course rubric.
    - If question/student_answer is absent: generates a technical interview question.
    """
    initial_state: TutorState = {
        "mode": "interview",
        "topic": topic or "LLM Engineering",
        "question": question or "",
        "student_answer": student_answer or "",
        "retry_count": 0
    }
    result = workflow.run(initial_state)

    return {
        "feedback_or_question": result.get("final_answer", ""),
        "intent": "interview_practice",
        "sources": result.get("sources", [])
    }
