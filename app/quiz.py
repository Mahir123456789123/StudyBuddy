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
    # Strip markdown code fences if present (```json ... ``` or ``` ... ``` or unclosed ```json ...)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", cleaned, re.IGNORECASE)
    if match and ("[" in match.group(1) or "{" in match.group(1)):
        cleaned = match.group(1).strip()
    return cleaned


def _extract_objects_from_broken_json(text: str) -> List[Dict[str, Any]]:
    """Recovers completed JSON objects from truncated or partially broken responses."""
    items: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        start_brace = text.find("{", idx)
        if start_brace == -1:
            break

        depth = 0
        end_brace = -1
        in_string = False
        escape = False

        for i in range(start_brace, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_brace = i
                        break

        if end_brace != -1:
            obj_str = text[start_brace:end_brace + 1]
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and "question" in obj and "options" in obj:
                    items.append(obj)
            except Exception:
                pass
            idx = end_brace + 1
        else:
            break

    return items


def validate_quiz_json(raw_text: str) -> List[QuizItem]:
    """
    Parses LLM generation, validates fields, ensures 4 options and valid answer_index.
    Drops malformed items. Returns a list of validated QuizItem objects.
    """
    cleaned_json = extract_json_block(raw_text)
    data = None
    try:
        data = json.loads(cleaned_json)
    except Exception as e:
        logger.warning(f"Direct JSON parse failed: {e}. Attempting object recovery...")
        recovered = _extract_objects_from_broken_json(cleaned_json)
        if recovered:
            data = recovered
        else:
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
        # Validate that all options are non-empty and all 4 are distinct
        if any(len(opt) == 0 for opt in options):
            continue
        if len(set(opt.lower() for opt in options)) < 4:
            continue

        try:
            answer_idx = int(raw_idx)
            if not (0 <= answer_idx <= 3):
                continue
        except (TypeError, ValueError):
            continue

        # Prevent duplicate questions or identical option sets across the quiz
        if any(item.question.lower() == q_text.lower() for item in validated_items):
            continue
        opt_signature = tuple(sorted(opt.lower() for opt in options))
        if any(tuple(sorted(o.lower() for o in item.options)) == opt_signature for item in validated_items):
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


TOPIC_QUIZ_POOLS: Dict[str, List[Dict[str, Any]]] = {
    "rag": [
        {
            "question": "How does the bi-encoder architecture in sentence-transformers/all-MiniLM-L6-v2 process queries and documents?",
            "options": [
                "By concatenating query and documents into a single cross-attention matrix",
                "By independently encoding queries and documents into 384-dimensional dense vectors",
                "By performing sparse BM25 term frequency hashing on CPU registers",
                "By computing exact regex pattern matches across document tokens"
            ],
            "answer_index": 1,
            "explanation": "Bi-encoders encode queries and candidate text passages into dense vectors independently, enabling fast vector similarity lookup.",
            "source": "lecture_slides_rag_and_embeddings.pdf"
        },
        {
            "question": "Why is RecursiveCharacterTextSplitter configured with a chunk overlap (e.g. chunk_size=500, chunk_overlap=80)?",
            "options": [
                "To prevent sentences and concepts from being abruptly truncated across boundary edges",
                "To encrypt sensitive student information inside chunk vectors",
                "To reduce the dimension of the embedding space from 768 to 384",
                "To eliminate the need for an embedding model entirely"
            ],
            "answer_index": 0,
            "explanation": "Chunk overlap maintains semantic context across boundaries, preventing essential factual phrases from being sliced in half.",
            "source": "lecture_slides_rag_and_embeddings.pdf"
        },
        {
            "question": "What is the primary role of dense retrieval in mitigating LLM hallucinations in StudyBuddy?",
            "options": [
                "It grounds completions in retrieved authoritative course passages rather than parametric memory",
                "It restricts the model to 1-bit integer quantization",
                "It executes rule-based sentiment classification on questions",
                "It bypasses the need for prompt context templates"
            ],
            "answer_index": 0,
            "explanation": "RAG supplies factual grounding data directly into the LLM context prompt, preventing ungrounded parametric hallucinations.",
            "source": "lecture_slides_rag_and_embeddings.pdf"
        },
        {
            "question": "In dense retrieval evaluation, what does the 'hit-rate' metric measure?",
            "options": [
                "The percentage of test questions where the ground-truth document chunk appears in the top-k retrieved results",
                "The clock frequency of GPU CUDA cores during similarity search",
                "The number of network packets dropped during HTTP streaming",
                "The total number of parameters fine-tuned by LoRA"
            ],
            "answer_index": 0,
            "explanation": "Hit-rate measures whether the correct gold source chunk is present within the retrieved top-k context passages.",
            "source": "lecture_slides_rag_and_embeddings.pdf"
        },
        {
            "question": "Why is the FAISS vector index built once at ingestion time rather than dynamically per request?",
            "options": [
                "To prevent catastrophic multi-second latencies and redundant CPU/GPU embedding computations on every query",
                "Because FAISS indices cannot be stored on disk",
                "Because Python does not allow reading files inside API routes",
                "To prevent the embedding model from overfitting"
            ],
            "answer_index": 0,
            "explanation": "Pre-building and loading the index into memory once at startup ensures sub-50ms vector similarity lookups during API calls.",
            "source": "lecture_slides_rag_and_embeddings.pdf"
        }
    ],
    "faiss": [
        {
            "question": "How does faiss.IndexFlatL2 compute nearest neighbors during similarity search?",
            "options": [
                "By performing an exhaustive, exact Euclidean L2 distance calculation across all stored vectors",
                "By randomly picking chunks using Monte Carlo sampling",
                "By hashing strings into integer bloom filters",
                "By training an autoencoder on vector clusters"
            ],
            "answer_index": 0,
            "explanation": "IndexFlatL2 calculates exact pairwise Euclidean distance (L2 norm) between query vector and every vector in the index.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "When scaling a vector store to millions of vectors, why would an inverted file index (IndexIVFFlat) be preferred over IndexFlatL2?",
            "options": [
                "It clusters vectors into Voronoi cells to search only a subset of centroids, dramatically accelerating search latency",
                "It eliminates the need for vector embeddings",
                "It uses 0 bytes of RAM",
                "It guarantees 100% exact nearest neighbor recall on every query"
            ],
            "answer_index": 0,
            "explanation": "IndexIVFFlat uses inverted lists and Voronoi cells to prune search space, trading small recall loss for major latency gains at scale.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "Why must vector embeddings be normalized to unit length (L2 norm = 1.0) when performing cosine similarity with IndexFlatIP?",
            "options": [
                "Because inner product (dot product) is mathematically identical to cosine similarity only for unit vectors",
                "Because FAISS crashes if vector values exceed 100",
                "Because unit vectors reduce the embedding dimension from 384 to 128",
                "To convert floating point numbers into integers"
            ],
            "answer_index": 0,
            "explanation": "For normalized unit vectors, the inner product dot(u, v) equals cosine(u, v) because ||u|| * ||v|| = 1.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "What is the vector dimensionality produced by all-MiniLM-L6-v2 and stored in StudyBuddy's FAISS index?",
            "options": [
                "384 dimensions",
                "768 dimensions",
                "1536 dimensions",
                "4096 dimensions"
            ],
            "answer_index": 0,
            "explanation": "sentence-transformers/all-MiniLM-L6-v2 outputs dense embedding vectors of dimension 384.",
            "source": "glossary_of_llm_terms.docx"
        },
        {
            "question": "In LangChain FAISS integration, what is the purpose of allow_dangerous_deserialization=True when loading an index?",
            "options": [
                "It allows unpickling the trusted local metadata dictionary stored alongside the FAISS binary file",
                "It disables memory protections on the host operating system",
                "It grants root access to incoming network clients",
                "It allows arbitrary code execution from remote web servers"
            ],
            "answer_index": 0,
            "explanation": "allow_dangerous_deserialization=True is required by LangChain to load local index pickle metadata that you generated yourself.",
            "source": "course_notes_orchestration_and_apis.txt"
        }
    ],
    "lora": [
        {
            "question": "What is the primary role of the low-rank matrices A and B in LoRA?",
            "options": [
                "To replace the entire weight matrix W with a random projection",
                "To decompose the weight update Delta W into (alpha/r)*B*A to reduce trainable parameters",
                "To quantize activations from 16-bit float into 4-bit NormalFloat",
                "To index embeddings inside a FAISS L2 flat vector index"
            ],
            "answer_index": 1,
            "explanation": "LoRA freezes the base weights W0 and represents the update Delta W as (alpha/r)*B*A, dramatically reducing trainable parameters and optimizer memory.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "In LoRA, how are matrices A and B initialized at the start of fine-tuning?",
            "options": [
                "Both A and B are initialized with identical random Gaussian weights",
                "Matrix A is initialized with random Gaussian noise and B is initialized to zero",
                "Matrix A is initialized to zero and B is initialized to identity",
                "Both A and B are initialized to 1.0"
            ],
            "answer_index": 1,
            "explanation": "Initializing B to zero ensures Delta W = B*A = 0 at the start of training, preserving the exact original pre-trained model behavior.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "What happens when rank r is increased in LoRA (e.g. from r=4 to r=16)?",
            "options": [
                "Trainable parameter count increases linearly with r, offering higher expressive capacity at increased memory cost",
                "The base model permanently loses pre-trained knowledge",
                "Inference speed slows down by 10x after weights are merged",
                "Quantization precision decreases from 4-bit to 2-bit"
            ],
            "answer_index": 0,
            "explanation": "Increasing r provides more bottleneck capacity for adaptation, but increases trainable parameter count and memory consumption.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "Why does LoRA incur zero inference latency overhead after deployment?",
            "options": [
                "Because the adapter weights Delta W = (alpha/r)*B*A can be directly folded and added into the base weights W0 before serving",
                "Because LoRA converts all Transformer layers to C++ binaries",
                "Because LoRA disables all attention projection layers during inference",
                "Because LoRA operates strictly outside GPU memory"
            ],
            "answer_index": 0,
            "explanation": "At inference time, the adapter Delta W can be directly merged into base matrix W0 = W0 + Delta W, resulting in standard dense inference with zero added latency.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "In Qwen2.5-0.5B, which target modules are standardly adapted with LoRA?",
            "options": [
                "Attention projection layers such as q_proj and v_proj",
                "Only the tokenizer vocabulary embedding table",
                "Only the final RMSNorm layer",
                "Only the positional rotary embedding vectors"
            ],
            "answer_index": 0,
            "explanation": "Query and value projection matrices (q_proj, v_proj) are the primary attention modules targeted in parameter-efficient fine-tuning.",
            "source": "lab_manual_fine_tuning.pdf"
        }
    ],
    "qlora": [
        {
            "question": "What data type innovation does QLoRA introduce for storing frozen base model weights?",
            "options": [
                "Standard uniform INT4 integer representation",
                "NormalFloat4 (NF4), an information-theoretically optimal quantile quantization for normally distributed weights",
                "Unsigned 8-bit integers (uint8)",
                "IEEE 754 64-bit double precision float"
            ],
            "answer_index": 1,
            "explanation": "NF4 matches the empirical normal distribution of neural network weights, achieving lower quantization error than uniform INT4.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "What is the purpose of Double Quantization (DQ) in QLoRA?",
            "options": [
                "To quantize activations twice during forward pass",
                "To quantize the quantization constants themselves, saving approximately 0.37 bits per parameter",
                "To double the number of trainable LoRA adapters in attention blocks",
                "To compress the FAISS index to 1-bit vectors"
            ],
            "answer_index": 1,
            "explanation": "Double Quantization quantizes the first-level quantization scale factors, reducing quantization memory overhead from 0.5 bits/param to 0.127 bits/param.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "What role does bnb_4bit_compute_dtype=torch.float16 play in QLoRA?",
            "options": [
                "De-quantizes 4-bit weights into 16-bit float on-the-fly for matrix multiplication computations during forward and backward passes",
                "Permanently casts model weights on disk to FP16",
                "Disables GPU CUDA acceleration",
                "Forces training to execute on 32-bit CPU registers"
            ],
            "answer_index": 0,
            "explanation": "Weights are stored in 4-bit NF4 to save VRAM, but computation is performed in 16-bit float during matrix multiplications.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "Why is prepare_model_for_kbit_training() called before attaching LoRA adapters to a quantized model?",
            "options": [
                "It casts layer normalization and output layers to 32-bit float and enables gradient checkpointing for stable training",
                "It deletes unquantized layers from disk",
                "It downloads pre-trained LoRA checkpoints from Hugging Face",
                "It converts PyTorch tensors into ONNX models"
            ],
            "answer_index": 0,
            "explanation": "prepare_model_for_kbit_training stabilizes 4-bit backpropagation by casting precision-sensitive operations to float32.",
            "source": "lab_manual_fine_tuning.pdf"
        },
        {
            "question": "Roughly how much GPU VRAM does a 4-bit quantized Qwen2.5-0.5B base model occupy compared to FP16?",
            "options": [
                "Roughly 70-75% less memory (around 300-400MB vs ~1.2GB in FP16)",
                "Zero memory reduction",
                "10x more memory due to quantization overhead",
                "Exactly 50% more memory"
            ],
            "answer_index": 0,
            "explanation": "4-bit quantization reduces base model weight footprint by roughly 4x compared to 16-bit precision.",
            "source": "lab_manual_fine_tuning.pdf"
        }
    ],
    "langgraph": [
        {
            "question": "In LangGraph, how is state managed across graph execution nodes?",
            "options": [
                "Using a TypedDict or Pydantic state schema passed and modified across nodes",
                "Through global OS environment variables",
                "Inside raw SQLite tables only",
                "Using HTTP cookie headers"
            ],
            "answer_index": 0,
            "explanation": "LangGraph workflows pass a centralized TypedDict or Pydantic state object to each node, which returns updated state keys.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "How do bounded retries protect a LangGraph RAG workflow from latency runaway?",
            "options": [
                "By checking retry_count < 1 on conditional edges to limit query rewriting to at most one retry",
                "By terminating the server process after 10 seconds",
                "By dropping all incoming network connections",
                "By setting the model temperature to zero"
            ],
            "answer_index": 0,
            "explanation": "Conditional edges enforce retry_count < 1, routing uncorrectable queries to fallback after a single retry.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "What distinguishes cyclic LangGraph workflows from linear RAG chains?",
            "options": [
                "The ability to execute self-correction loops, dynamic intent branching, and verification steps",
                "LangGraph requires GPU clusters while linear chains run on CPU",
                "LangGraph cannot output text",
                "Linear chains cannot use vector databases"
            ],
            "answer_index": 0,
            "explanation": "LangGraph supports directed state graphs with cycles, conditional transitions, and self-correction loops.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "In StudyBuddy's LangGraph routing, what happens when intent is classified as out_of_scope?",
            "options": [
                "It immediately transitions to node_fallback without wasting retrieval or LLM drafting cycles",
                "It attempts 5 recursive web searches",
                "It crashes the FastAPI process",
                "It deletes the FAISS vector index"
            ],
            "answer_index": 0,
            "explanation": "Out-of-scope routing avoids expensive retriever queries and LLM token generations by sending polite refusal text immediately.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "What is the primary role of node_review_answer in the StudyBuddy LangGraph pipeline?",
            "options": [
                "To verify that draft responses do not contain INSUFFICIENT_CONTEXT refusal markers before answering",
                "To grade multiple-choice quizzes with machine learning models",
                "To compile Python code to bytecode",
                "To re-rank FAISS vector chunks with cross-attention"
            ],
            "answer_index": 0,
            "explanation": "The review node inspects LLM drafts, triggering query rewrite if the draft contains refusal markers or lacks context grounding.",
            "source": "course_notes_orchestration_and_apis.txt"
        }
    ],
    "fastapi": [
        {
            "question": "Why should heavy components like FAISS indexes and embedding models be loaded in a FastAPI lifespan handler?",
            "options": [
                "To initialize them once at server startup in app.state rather than reloading per request",
                "Because FastAPI prohibits loading models outside HTML templates",
                "To avoid creating Python virtual environments",
                "Because Windows cannot load FAISS inside endpoints"
            ],
            "answer_index": 0,
            "explanation": "The lifespan handler caches heavy objects once in memory, avoiding catastrophic multi-second latencies per request.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "Why is multiple-choice quiz grading performed deterministically in application code rather than by an LLM?",
            "options": [
                "To guarantee 100% grading objectivity and eliminate LLM hallucination or grading bias",
                "Because LLMs cannot process numbers",
                "Because grading requires GPU CUDA tensors",
                "Because JSON cannot be sent over HTTP"
            ],
            "answer_index": 0,
            "explanation": "Deterministic index comparison in Python code guarantees exact scoring with zero hallucination.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "Why is answer_index stripped from the client payload when /quiz returns questions to the frontend?",
            "options": [
                "To prevent students from inspecting the network tab or browser DevTools to see the answer keys",
                "Because JSON does not support integer numbers",
                "Because answer keys cannot be transmitted over HTTP",
                "To reduce response size by 99%"
            ],
            "answer_index": 0,
            "explanation": "Quiz security requires storing answer keys strictly server-side in session memory, returning only questions and options to the client.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "How does Pydantic input validation protect FastAPI endpoints against malicious or malformed requests?",
            "options": [
                "By rejecting empty/whitespace inputs, enforcing string length limits, and validating field types before endpoint execution",
                "By disabling the Windows firewall",
                "By encrypting the entire hard drive",
                "By converting all strings into binary floats"
            ],
            "answer_index": 0,
            "explanation": "Pydantic request schemas validate data types, max lengths, and constraints, rejecting malformed requests with 422 before handlers run.",
            "source": "course_notes_orchestration_and_apis.txt"
        },
        {
            "question": "What HTTP status code should be returned if a student attempts to grade an expired or non-existent quiz ID?",
            "options": [
                "404 Not Found",
                "200 OK with empty body",
                "500 Internal Server Error",
                "301 Moved Permanently"
            ],
            "answer_index": 0,
            "explanation": "When an entity or session identifier is not present in storage, REST best practices specify HTTP 404 Not Found.",
            "source": "course_notes_orchestration_and_apis.txt"
        }
    ]
}


def get_topic_fallback_quiz(topic: str, needed: int = 5) -> List[QuizItem]:
    """
    Returns high-quality, verified course multiple-choice questions for the requested topic.
    Guarantees the exact requested number of questions without returning static single placeholders.
    """
    if needed <= 0:
        return []

    t_low = (topic or "").lower()

    if any(k in t_low for k in ["rag", "retriev", "dense", "bi-encoder", "minilm", "embedding", "chunk"]):
        pool = TOPIC_QUIZ_POOLS["rag"]
    elif any(k in t_low for k in ["faiss", "index", "vector", "flatl2", "ivfflat"]):
        pool = TOPIC_QUIZ_POOLS["faiss"]
    elif any(k in t_low for k in ["qlora", "quantiz", "nf4", "double quant", "bitsandbytes"]):
        pool = TOPIC_QUIZ_POOLS["qlora"]
    elif any(k in t_low for k in ["langgraph", "orchestrat", "graph", "workflow", "state", "retry"]):
        pool = TOPIC_QUIZ_POOLS["langgraph"]
    elif any(k in t_low for k in ["fastapi", "api", "production", "endpoint", "pydantic", "lifespan"]):
        pool = TOPIC_QUIZ_POOLS["fastapi"]
    elif any(k in t_low for k in ["lora", "peft", "fine-tun", "adapter", "rank", "alpha"]):
        pool = TOPIC_QUIZ_POOLS["lora"]
    else:
        # Out of scope topic (e.g. "mahir", "weather", "recipe")
        pool = [
            {
                "question": f"The topic '{topic}' is not in the LLM Engineering course documents. Which of the following is an authorized course topic?",
                "options": [
                    "Low-Rank Adaptation (LoRA) and QLoRA",
                    "Dense Retrieval with all-MiniLM-L6-v2 and FAISS",
                    "LangGraph Stateful Workflows & Bounded Retries",
                    "All of the above"
                ],
                "answer_index": 3,
                "explanation": f"'{topic}' is outside the LLM engineering syllabus. Authorized course topics include LoRA/QLoRA, RAG/Embeddings, FAISS, and LangGraph.",
                "source": "Course Syllabus"
            },
            {
                "question": "Which of the following topics is part of our LLM Engineering course curriculum?",
                "options": [
                    "NormalFloat4 (NF4) Quantization & QLoRA",
                    "Culinary baking recipes",
                    "Real estate mortgage underwriting",
                    "Automobile engine diagnostics"
                ],
                "answer_index": 0,
                "explanation": "Our course covers LLM fine-tuning, quantization, dense retrieval, and graph orchestration.",
                "source": "Course Syllabus"
            }
        ]

    # Convert dictionaries to QuizItem
    items: List[QuizItem] = [QuizItem(**d) for d in pool]

    # If user needs more items than pool length, combine with other course questions
    if len(items) < needed:
        for other_key in ["lora", "qlora", "rag", "faiss", "langgraph", "fastapi"]:
            for d in TOPIC_QUIZ_POOLS[other_key]:
                q_item = QuizItem(**d)
                if not any(it.question == q_item.question for it in items):
                    items.append(q_item)
                if len(items) >= needed:
                    break
            if len(items) >= needed:
                break

    return items[:needed]
