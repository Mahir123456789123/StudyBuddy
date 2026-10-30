# StudyBuddy: Course Tutor & Technical Interview Preparation Assistant

## 1. Project Summary
StudyBuddy is an end-to-end, course-grounded conversational AI tutor and technical interview preparation assistant designed for an advanced LLM Engineering curriculum. The system leverages course materials—including laboratory manuals, exported lecture slide decks, structured instructor notes, and a comprehensive glossary—to provide accurate, grounded responses with strict file and page citations.

### Core Student Capabilities:
1. **Concept Explanations**: Ask conceptual questions regarding LLM engineering (LoRA, QLoRA, Quantization, Vector Search, FAISS, Graph Orchestration) and receive answers strictly grounded in retrieved passages with source metadata (filename and page number).
2. **Quiz Generation**: Request multiple-choice quizzes (1 to 5 questions) for any course topic, generating structured questions with 4 distinct options, verified answer keys, and pedagogical explanations.
3. **Automated Quiz Grading**: Submit student answers to receive deterministic, objective grading, immediate explanations, and direct excerpts from the authoritative course texts.
4. **Interactive Interview Practice**: Practice technical interview questions under a mock interviewer setting, receiving structured feedback against an evaluation rubric constructed directly from course literature.

### Foundational Design Principle:
> **Fine-tuning teaches HOW to respond** (tutor persona, pedagogical clarity, strict JSON schema output, fallback phrasing);
> **Retrieval-Augmented Generation (RAG) supplies WHAT to respond with** (authoritative technical facts, hyperparameter details, course citations).

---

## 2. System Architecture & Components

```
+-----------------------------------------------------------------------------------+
|                                  Frontend UI                                      |
|                       (Vanilla HTML5, CSS3, JS Single Page)                       |
+-----------------------------------------------------------------------------------+
                                          |
                                    HTTP Requests
                                          v
+-----------------------------------------------------------------------------------+
|                             FastAPI Web Service                                   |
|   - Lifespan State Initialization (Embeddings, FAISS Index, Graph, Backends)      |
|   - Pydantic Validation, Latency Interceptor, Structured Audit Logging             |
|   - Endpoints: GET /health, POST /ask, POST /quiz, POST /quiz/grade, POST /interview |
+-----------------------------------------------------------------------------------+
                                          |
                                   State Invocations
                                          v
+-----------------------------------------------------------------------------------+
|                           LangGraph Orchestrator                                  |
|                                                                                   |
|          [START] ---> [classify_intent] ---> [retrieve]                           |
|                              |                                                    |
|                    (Conditional Branching)                                        |
|                     /        |         \         \                                |
|        explain_concept   quiz_request  interview  out_of_scope                    |
|             |                |             |              |                       |
|       [draft_answer]  [generate_quiz]  [interview_node]   |                       |
|             |                |                            |                       |
|      [review_answer]  [validate_quiz]                     |                       |
|             \               /                             |                       |
|       (ok? / retry < 1? / fallback)                       |                       |
|             /                \                            |                       |
|      [rewrite_query] ---> [retrieve]                  [fallback]                  |
|                                                           |                       |
|                                                        [END]                      |
+-----------------------------------------------------------------------------------+
         |                                                 |
         v                                                 v
+---------------------------+              +-------------------------------+
|      FAISS Vector DB      |              |          LLM Backend          |
|  - all-MiniLM-L6-v2       |              |  - mock (Deterministic Test)  |
|  - IndexFlatL2 Persistence|              |  - hf_local (Qwen2.5 + LoRA)  |
|  - Multi-source metadata  |              |  - ollama (Local endpoint)    |
+---------------------------+              |  - cloud (OpenAI-compatible)  |
                                           +-------------------------------+
```

---

## 3. Experimental Curriculum Mapping

| Experiment ID | Focus Area | Artifacts / Implementation |
|---|---|---|
| **Experiment 1** | Baseline Fine-Tuning Pipeline | `notebooks/01_finetune_pipeline_distilgpt2.ipynb` |
| **Experiment 2** | LoRA Hyperparameter Study (Rank, LR, Epochs) | `notebooks/02_lora_hyperparameter_study.ipynb` |
| **Experiment 3** | Quantization & QLoRA Memory Profiling (FP16 vs INT8 vs NF4) | `notebooks/03_quantization_and_qlora.ipynb` |
| **Experiment 4** | Raw FAISS RAG Baseline (No LangChain) | `notebooks/04_basic_faiss_rag.ipynb` |
| **Experiment 5** | Production Document Ingestion & Vector Retrieval | `app/ingestion.py`, `app/retriever.py` |
| **Experiment 6** | Stateful LangGraph Orchestration & Self-Correction | `app/graph.py`, `app/quiz.py` |
| **Experiment 7** | Production FastAPI Service & Observability | `app/main.py`, `app/logging_utils.py` |
