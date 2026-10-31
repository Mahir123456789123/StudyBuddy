# StudyBuddy: LLM Engineering Course Tutor & Technical Interview Assistant

StudyBuddy is a comprehensive, production-oriented course tutor and technical interview preparation assistant developed for an advanced LLM Engineering curriculum. The system’s foundational knowledge base is the course literature itself: laboratory manuals (PDF), lecture slide decks (PDF), student and instructor notes (TXT), and a technical glossary (DOCX).

---

## 1. Core Architectural Principle

> **Fine-Tuning teaches HOW to respond** (tutor tone, concise structure, strict multiple-choice quiz JSON schemas, rubric formatting, and refusal markers).  
> **Retrieval-Augmented Generation (RAG) supplies WHAT to respond with** (factual accuracy, authoritative mathematical formulas, hyperparameters, and verifiable file and page citations).

---

## 2. System Features & Student Capabilities

1. **Concept Explanations (`POST /ask`)**: Ask conceptual engineering questions and receive concise explanations grounded in course documents with exact filename and page citations.
2. **Quiz Generation (`POST /quiz`)**: Request 1 to 5 multiple-choice questions on any course topic. Generates strict JSON schemas with 4 options and server-side key verification.
3. **Deterministic Quiz Grading (`POST /quiz/grade`)**: Multiple-choice grading runs in Python code with zero LLM hallucination, returning instant scores, explanations, and ground-truth source passages.
4. **Technical Interview Practice (`POST /interview`)**: Receive challenging interview questions and submit candidate answers evaluated against reference rubrics (Verdict, What Was Good, What Was Missing, Model Answer).
5. **Observability & Security**: Request latency tracking (`X-Response-Time-ms`), structured JSON audit logs with sensitive input truncation, and defense-in-depth against prompt injection.

---

## 3. Repository Structure

```
studybuddy/
  README.md                              # Complete setup, architecture & viva guide
  PROJECT_SPEC.md                        # Formal engineering curriculum specification
  requirements.txt                       # Project dependencies
  pytest.ini                             # Pytest configuration
  .env.example                           # Template environment settings
  .gitignore                             # Version control exclusions
  data/
    raw/                                 # 19 canonical course documents (PDF, TXT, DOCX)
      Lecture1LLM_FineTuning_PartA_Foundations_.pdf
      Lecture2Introduction to Fine-Tuning LLMs.pdf
      Lecture3LLM_FineTuning_Process_and_Taxonomy_.pdf
      Lecture4LLM_FineTuning_PEFT_LoRA_QLoRA_.pdf
      ... [17 Lecture PDFs, Student Notes, TXT Notes, DOCX Glossary]
    processed/                           # Chunk inspection dump
      chunks_dump.json
    finetune/
      train.jsonl                        # 600 training pairs grounded in real course lectures
      review_sample.csv                  # 50 random rows for human verification
      eval_questions.jsonl               # 20 strictly held-out unseen evaluation questions (0 overlap)
  notebooks/
    01_finetune_pipeline_distilgpt2.ipynb # Exp 1: Baseline pipeline
    02_lora_hyperparameter_study.ipynb    # Exp 2: LoRA hyperparameter optimization
    03_quantization_and_qlora.ipynb       # Exp 3: Precision profiling & QLoRA
    04_basic_faiss_rag.ipynb              # Exp 4: Raw FAISS RAG from scratch
  app/
    __init__.py
    config.py                            # Pydantic environment configuration
    logging_utils.py                     # Structured audit logging & privacy truncation
    prompts.py                           # Centralized system and task prompt templates
    llm_backends.py                      # 4 pluggable backends (mock, hf_local, ollama, cloud)
    ingestion.py                         # PDF/TXT/DOCX ingestion & FAISS indexer (Exp 5)
    retriever.py                         # Similarity search & metadata citations
    quiz.py                              # MCQ schema validation, store & deterministic grading
    graph.py                             # Stateful LangGraph workflow with bounded retry (Exp 6)
    tutor_rag.py                         # Public orchestration facade
    main.py                              # Production FastAPI web application (Exp 7)
  eval/
    retrieval_eval.py                    # Top-k hit-rate & chunk size comparison
    run_comparison.py                    # Four-way model benchmark & observation tables
    results/                             # CSV benchmarks & markdown summaries
  tests/
    test_api.py                          # FastAPI endpoint integration tests
    test_validation.py                   # Pydantic boundary and length rejection tests
    test_graph_routing.py                # LangGraph intent transitions & injection tests
    test_quiz_parsing.py                 # Quiz parsing & deterministic grading tests
  frontend/
    index.html                           # Single-page interactive student UI
  report/
    observation_tables.md                # Consolidated experimental tables for viva
  docs/
    graph.md                             # Compiled LangGraph Mermaid architecture diagram
```

---

## 4. Quickstart Guide (Laptop / CPU-Only Setup)

### 4.1 Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 4.2 Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd StudyBuddy

# 2. Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 4.3 Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration uses `LLM_BACKEND=mock` which allows the entire application, test suite, and evaluation scripts to run on a standard CPU laptop without GPUs, Ollama, or paid cloud APIs.

### 4.4 Ingestion & Index Creation

```bash
# Ingest PDF, TXT, DOCX documents into FAISS vector index
python -m app.ingestion --rebuild
```

### 4.5 Run Test Suite

```bash
python -m pytest -v
```
All 23 integration and unit tests will execute and pass offline.

### 4.6 Launch Web Service

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Web UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- OpenAPI / Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 5. LLM Backend Configurations

StudyBuddy supports 4 pluggable backends selected via the `LLM_BACKEND` environment variable:

### 1. `mock` (Default)
Deterministic fake responses. Fast, reproducible, and requires zero external network calls or GPU VRAM.

### 2. `hf_local` (Hugging Face Transformers + LoRA Adapter)
Loads the base model `Qwen/Qwen2.5-0.5B-Instruct` and attaches the trained QLoRA adapter:
```env
LLM_BACKEND=hf_local
HF_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-0.5B-Instruct
LORA_ADAPTER_PATH=models/adapter/
```
> **Adapter Placement**: After training in Colab (Notebook 03), download the `./qlora_adapter` directory and place its contents into `models/adapter/` (`adapter_config.json` and `adapter_model.safetensors`).

### 3. `ollama` (Local Ollama Endpoint)
Connects to a running local Ollama instance:
```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:0.5b
```

### 4. `cloud` (OpenAI-Compatible Free/Commercial API)
Connects to OpenAI, Groq, OpenRouter, or Gemini OpenAI endpoints:
```env
LLM_BACKEND=cloud
CLOUD_API_BASE=https://api.openai.com/v1
CLOUD_API_KEY=sk-...
CLOUD_MODEL_NAME=gpt-4o-mini
```

---

## 6. Reproducing All Curriculum Evaluations

To reproduce every evaluation table required for viva examination:

```bash
# 1. Verify dataset isolation (ensures 0 overlap between training and eval sets)
python scripts/verify_dataset_overlap.py

# 2. Run retrieval Hit@k and chunk-size comparison (300 vs 500 vs 800)
python eval/retrieval_eval.py

# 3. Run Four-Way Model Benchmark & LangGraph Behavior Study
python eval/run_comparison.py
```

Generated tables are exported to `eval/results/` and summarized in `report/observation_tables.md`:
- `eval/results/four_way_comparison_summary.csv`
- `eval/results/retrieval_hit_rates.csv`
- `eval/results/retrieval_chunk_size_comparison.csv`
- `eval/results/graph_behaviour_table.csv`
- `eval/results/api_test_observation_table.csv`

---

## 7. Experimental Curriculum Summary

| Exp | Description | Key Findings / Winning Configuration |
|---|---|---|
| **Exp 1** | Baseline DistilGPT-2 Pipeline | Full causal LM fine-tuning yields poor instruction following; confirms need for instruction-tuned base model (`Qwen2.5-0.5B-Instruct`). |
| **Exp 2** | LoRA Hyperparameter Study | **Run C Winner**: $\text{lr}=2\times 10^{-4}$, $\text{epochs}=2$, $\text{rank } r=8$, $\alpha=16$. Trains 0.109% parameters (train loss: 3.6420) with highest holdout quality score (2.55 / 5.0 across all 20 eval questions, selected via code) and saved in `models/adapter/`. Run B achieved 2.52 with half the parameter count ($r=4$, 0.055%), effectively tied with a negligible 0.03 gap. Run D achieved lowest loss (3.2009) with 2.45 quality. |
| **Exp 3** | Quantization & QLoRA | Profiled on NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4). NF4 with Double Quantization cuts allocated VRAM by **52.8%** (from 942.3 MB down to 444.6 MB) with 3929.9 ms generation latency. |
| **Exp 4** | Raw FAISS RAG Baseline | Validates dense bi-encoder cosine/L2 mapping with native `faiss.IndexFlatL2` with `faiss-cpu`, achieving 100% precision on in-domain course chunks. |
| **Exp 5** | Production Ingestion & Index | Multi-format loader (17 lecture PDFs, student notes, DOCX glossary, TXT notes) indexing 918 chunks into FAISS vector store. Chunk Size Trade-offs: Chunk 800 achieved highest Hit@3 (85.7%, 12/14) with 600 broad chunks; Chunk 300 achieved highest Hit@1 (42.9%, 6/14) with 1,460 granular chunks; Chunk 500 achieved 64.3% Hit@3 (9/14) with 918 chunks. In this small curriculum set ($N=14$ in-scope questions), each question represents ~7.1 percentage points ($\pm 7.1\%$). |
| **Exp 6** | LangGraph Orchestration | Self-correction loop with bounded retry ($\text{retries} < 1$) and prompt-injection defense. |
| **Exp 7** | Production FastAPI Service | Sub-25ms P95 API overhead, deterministic quiz grading, structured JSON audit logging without credential leaks. |

---

## 8. Limitations and Production Gaps (Viva Discussion)

While StudyBuddy satisfies all pedagogical and architectural requirements, the following technical limitations are acknowledged for viva defense:

1. **Small-Model Weaknesses**: The base model `Qwen2.5-0.5B-Instruct` is compact (490M parameters). While fast and memory-efficient, its complex logical multi-hop reasoning is lower than larger foundation models (7B or 70B).
2. **Fine-Tuning Helpfulness Bias on Fallbacks**: In our 4-way evaluation, Fine-Tuned + RAG achieves an 85.0% correct fallback rate on ambiguous/unsupported queries compared to 70.0% for the un-tuned Base Model and 70.0% for Base + RAG, while scoring 3.40 / 5.0 on genuine holdout outputs (decisively outperforming Base Model at 2.33 / 5.0). When queries fall outside course context, conversational training encourages constructive guidance; StudyBuddy's LangGraph pipeline directly remedies this by deterministically intercepting ungrounded intents at the graph root.
3. **Rule-Based Intent Classifier**: Intent routing relies on keyword heuristics for deterministic zero-latency routing. In high-entropy conversational settings, a small trained classifier or embedding-based semantic router would handle edge cases more gracefully.
4. **Prompt Injection in Documents**: While the tutor prompt treats context as reference material, sophisticated indirect prompt injection embedded within raw course materials remains a known risk in uncurated vector stores.
5. **In-Memory Quiz Session Store**: Active quiz sessions are held in application RAM with a capacity cap (`MAX_STORED_QUIZZES=200`). Distributed multi-worker deployments require persistent caching (e.g. Redis).
6. **Production Infrastructure Gaps**:
   - **Authentication & RBAC**: No user authentication (JWT/OAuth2) or role separation.
   - **Transport Security**: Running on local HTTP; production requires TLS termination via HTTPS/reverse proxy.
   - **Rate Limiting**: Missing token-bucket rate limiting against DoS requests.
   - **Observability**: Uses stdout structured JSON logging; enterprise production requires OpenTelemetry tracing and Prometheus metrics.
   - **Disaster Recovery**: Vector index is stored locally on disk; cloud storage (S3/GCS snapshots) needed for high availability.
