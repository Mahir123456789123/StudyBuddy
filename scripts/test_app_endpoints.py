import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app.graph import StudyBuddyWorkflow
from app.llm_backends import MockLLMBackend
from app.tutor_rag import ask_concept, generate_quiz_session, grade_quiz_session, practice_interview

wf = StudyBuddyWorkflow(llm=MockLLMBackend())

print("--- Test 1: Ask Concept ---")
res_ask = ask_concept("hi whats rag", wf)
print("Ask answer:", res_ask["answer"][:100])
print("Intent:", res_ask["intent"])

print("\n--- Test 2: Quiz RAG 5 Questions ---")
res_quiz = generate_quiz_session("RAG", 5, wf)
print("Quiz items count:", len(res_quiz["questions"]))
for q in res_quiz["questions"]:
    print("  Q:", q["question"][:70])

print("\n--- Test 3: Quiz mahir 1 Question ---")
res_mahir = generate_quiz_session("mahir", 1, wf)
print("Mahir items count:", len(res_mahir["questions"]))
for q in res_mahir["questions"]:
    print("  Q:", q["question"][:70])

print("\n--- Test 4: Interview Question for RAG ---")
res_int_rag = practice_interview("RAG", None, None, wf)
print("Interview Q:", res_int_rag["feedback_or_question"][:120])

print("\n--- Test 5: Interview Grading for 'okay' ---")
res_grade_ok = practice_interview(None, "Why is Pydantic used in FastAPI?", "okay", wf)
print(res_grade_ok["feedback_or_question"])

print("\n--- Test 6: Interview Grading for 'loss of quality' ---")
res_grade_loss = practice_interview(None, "What are the trade-offs of fine-tuning?", "loss of quality", wf)
print(res_grade_loss["feedback_or_question"])
