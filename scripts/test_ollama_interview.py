import sys, os
sys.path.insert(0, os.path.abspath("."))
from app.graph import StudyBuddyWorkflow
from app.llm_backends import OllamaBackend
from app.tutor_rag import practice_interview

backend = OllamaBackend(base_url="http://localhost:11434", model_name="qwen2.5:0.5b")
wf = StudyBuddyWorkflow(llm=backend)

print("--- Test Ollama Interview Question for RAG ---")
res = practice_interview("RAG and Dense Embeddings", None, None, wf)
print("Question:\n", res["feedback_or_question"])
