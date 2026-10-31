import sys, os
sys.path.insert(0, os.path.abspath("."))
from app.graph import StudyBuddyWorkflow
from app.llm_backends import OllamaBackend
from app.tutor_rag import ask_concept, generate_quiz_session, practice_interview

backend = OllamaBackend(base_url="http://localhost:11434", model_name="qwen2.5:0.5b")
wf = StudyBuddyWorkflow(llm=backend)

print("--- Test Ask Concept with Ollama ---")
res = ask_concept("hi whats rag", wf)
print("Answer:\n", res["answer"])
print("Sources:\n", res["sources"])
print("Intent:", res["intent"])
print("Retries:", res["retries"])
