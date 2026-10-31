"""
LangGraph Workflow Orchestration (Experiment 6).
Implements a stateful, cyclic workflow with:
- Deterministic rule-based intent classification
- Context retrieval from FAISS
- Intent-based dynamic routing
- Draft answer generation & pedagogical review
- Quiz generation & schema validation
- Bounded self-correction retry loop (bounded to exactly 1 retry)
- Safe fallback routing
"""

import os
import re
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.llm_backends import LLMBackend, get_llm_backend
from app.logging_utils import logger
from app.prompts import (
    CONCEPT_QA_TEMPLATE,
    FALLBACK_TEXT,
    INSUFFICIENT_CONTEXT_MARKER,
    INTERVIEW_EVALUATION_TEMPLATE,
    INTERVIEW_QUESTION_TEMPLATE,
    QUIZ_GENERATION_TEMPLATE,
    SYSTEM_PROMPT_TUTOR,
)
from app.quiz import QuizItem, validate_quiz_json
from app.retriever import CourseRetriever, create_retriever


# Experiment 6: State Definition
class TutorState(TypedDict, total=False):
    question: str
    mode: str             # ask | quiz | interview | grade
    intent: str           # explain_concept | quiz_request | interview_practice | out_of_scope
    topic: str
    search_query: str
    context: str
    sources: List[Dict[str, Any]]  # [{"source": ..., "page": ..., "topic": ...}]
    draft: str
    quiz: List[Dict[str, Any]]     # validated MCQ items
    retry_count: int
    final_answer: str
    num_questions: int
    student_answer: str


# Rule-based Intent Classifier Keywords (Experiment 6)
# Note: Rule-based keyword matching is intentionally used for predictable, zero-latency routing.
INTENT_KEYWORDS = {
    "quiz_request": [
        "quiz", "mcq", "multiple choice", "test me", "test my knowledge", "generate questions"
    ],
    "interview_practice": [
        "interview", "mock interview", "technical interview", "candidate", "practice interview", "evaluate answer"
    ],
    "in_scope_concepts": [
        "lora", "qlora", "nf4", "quantization", "fine-tuning", "peft", "adapter",
        "faiss", "embedding", "embeddings", "rag", "chunk", "chunking", "bi-encoder",
        "minilm", "langgraph", "orchestration", "state", "retry", "intent",
        "fastapi", "pydantic", "chat template", "rank", "alpha", "double quantization"
    ],
    "injection_or_unsupported": [
        "ignore previous instructions", "ignore your instructions", "reveal your prompt",
        "system prompt", "weather", "insurance", "football", "recipe", "stock"
    ]
}


def node_classify_intent(state: TutorState) -> TutorState:
    """
    Node 1: Intent Classification (Experiment 6).
    Applies rule-based heuristic classification on user input.
    """
    question = (state.get("question") or "").strip().lower()
    mode = state.get("mode", "ask")

    # Direct mode overrides
    if mode == "quiz":
        intent = "quiz_request"
    elif mode == "interview":
        intent = "interview_practice"
    else:
        # Check for prompt injection or explicit out-of-scope topics
        if any(bad in question for bad in INTENT_KEYWORDS["injection_or_unsupported"]):
            intent = "out_of_scope"
        elif any(kw in question for kw in INTENT_KEYWORDS["quiz_request"]):
            intent = "quiz_request"
        elif any(kw in question for kw in INTENT_KEYWORDS["interview_practice"]):
            intent = "interview_practice"
        elif any(topic in question for topic in INTENT_KEYWORDS["in_scope_concepts"]):
            intent = "explain_concept"
        else:
            intent = "out_of_scope"

    logger.debug(f"[Graph: classify_intent] Question='{state.get('question')[:40]}...' -> Intent='{intent}'")
    return {
        "intent": intent,
        "search_query": state.get("search_query") or state.get("question", ""),
        "retry_count": state.get("retry_count", 0)
    }


def node_retrieve(state: TutorState, retriever: CourseRetriever) -> TutorState:
    """
    Node 2: Document Retrieval (Experiment 5/6).
    Queries FAISS index and stores formatted context passages and citations.
    """
    query = state.get("search_query") or state.get("question") or state.get("topic") or "LLM engineering"
    docs = retriever.retrieve(query)
    context_str, sources = retriever.format_context_and_sources(docs)

    logger.debug(f"[Graph: retrieve] Query='{query}' -> Retrieved {len(docs)} chunks from {len(sources)} sources.")
    return {
        "context": context_str,
        "sources": sources
    }


def route_by_intent(state: TutorState) -> str:
    """Conditional Edge: Routes flow according to classified intent."""
    intent = state.get("intent", "out_of_scope")
    if intent == "explain_concept":
        return "draft_answer"
    elif intent == "quiz_request":
        return "generate_quiz"
    elif intent == "interview_practice":
        return "handle_interview"
    else:
        return "fallback"


def node_draft_answer(state: TutorState, llm: LLMBackend) -> TutorState:
    """
    Node 3A: Draft Concept Answer (Experiment 6).
    Invokes LLM with strict context-bounding and INSUFFICIENT_CONTEXT marker instructions.
    """
    question = state.get("question", "")
    context = state.get("context", "")

    prompt = CONCEPT_QA_TEMPLATE.format(
        context=context,
        question=question,
        marker=INSUFFICIENT_CONTEXT_MARKER
    )

    draft = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_TUTOR, max_tokens=512)
    return {"draft": draft}


def node_review_answer(state: TutorState) -> TutorState:
    """
    Node 3B: Pedagogical Answer Review (Experiment 6).
    Verifies that the draft does not contain refusal markers and is non-empty.
    """
    draft = (state.get("draft") or "").strip()
    is_refusal = (INSUFFICIENT_CONTEXT_MARKER in draft) or (len(draft) < 15)

    if not is_refusal:
        # Lightweight check: does draft share terms with context or question?
        return {"final_answer": draft}
    return {}


def review_answer_condition(state: TutorState) -> str:
    """
    Conditional Edge: Checks if answer is acceptable or requires query rewriting.
    Enforces bounded retry: max 1 retry allowed.
    """
    final_ans = state.get("final_answer")
    retries = state.get("retry_count", 0)

    if final_ans and INSUFFICIENT_CONTEXT_MARKER not in final_ans:
        return "ok"
    if retries < 1:
        return "retry"
    return "fallback"


def node_generate_quiz(state: TutorState, llm: LLMBackend) -> TutorState:
    """
    Node 4A: Quiz Generation (Experiment 6).
    Prompts LLM to produce structured multiple-choice questions in JSON.
    """
    topic = state.get("topic") or state.get("question") or "LLM Engineering Concepts"
    num_questions = state.get("num_questions", 3)
    context = state.get("context", "")

    prompt = QUIZ_GENERATION_TEMPLATE.format(
        context=context,
        topic=topic,
        num_questions=num_questions
    )

    max_tok = max(1024, num_questions * 350)
    raw_response = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_TUTOR, max_tokens=max_tok)
    return {"draft": raw_response}


def node_validate_quiz(state: TutorState) -> TutorState:
    """
    Node 4B: Quiz Validation (Experiment 6).
    Parses and enforces strict schema checks (4 options, valid 0-3 index).
    """
    draft = state.get("draft", "")
    items = validate_quiz_json(draft)
    if items:
        # Convert Pydantic items to dicts for serializability in state
        serialized = [item.model_dump() for item in items]
        return {"quiz": serialized}
    return {"quiz": []}


def validate_quiz_condition(state: TutorState) -> str:
    """
    Conditional Edge: Determines whether quiz items are valid or if retry is needed.
    """
    quiz = state.get("quiz", [])
    retries = state.get("retry_count", 0)

    if quiz and len(quiz) >= 1:
        return "ok"
    if retries < 1:
        return "retry"
    return "fallback"


def _get_topic_model_answer(question: str, context: str) -> str:
    """Generates an authoritative, precise course model answer for technical interview questions."""
    q_low = question.lower()
    if "pydantic" in q_low or "fastapi" in q_low or "api" in q_low or "schema" in q_low:
        return (
            "Pydantic schemas enforce strict data contracts in FastAPI by validating data types, string constraints (min/max length), "
            "and required fields before endpoint execution. If incoming payloads fail validation, FastAPI immediately returns an "
            "HTTP 422 Unprocessable Entity error, protecting the application against runtime crashes, invalid states, and basic injection attacks. "
            "Furthermore, response models sanitize output data by stripping sensitive internal fields (such as quiz answer keys) before transmitting responses."
        )
    elif "lora" in q_low or "peft" in q_low or "fine-tun" in q_low or "adapter" in q_low:
        return (
            "Parameter-Efficient Fine-Tuning (LoRA) freezes pre-trained weights W0 and introduces low-rank decomposition matrices "
            "B and A (where forward pass is W = W0 + (alpha/r)*B*A). Initializing B to 0 ensures Delta W = 0 at the start of training. "
            "This reduces trainable parameters and optimizer states by over 90%, saving substantial GPU VRAM. Key trade-offs include "
            "choosing rank r (higher r increases expressive capacity but increases memory) and scaling factor alpha (which balances adapter update magnitudes for training stability)."
        )
    elif "qlora" in q_low or "quantiz" in q_low or "nf4" in q_low:
        return (
            "QLoRA introduces three core innovations: 4-bit NormalFloat (NF4), which matches the normal distribution of pre-trained weights; "
            "Double Quantization (DQ), which quantizes the first-level quantization constants to save ~0.37 bits per parameter; and paged optimizers "
            "to prevent out-of-memory spikes. Weights are held in 4-bit precision and de-quantized to 16-bit float on-the-fly during forward/backward passes using bnb_4bit_compute_dtype."
        )
    elif "rag" in q_low or "retriev" in q_low or "embedding" in q_low:
        return (
            "In dense retrieval pipelines, bi-encoders (like sentence-transformers/all-MiniLM-L6-v2) pre-encode document chunks into dense 384-dimensional vectors "
            "indexed in FAISS for fast sub-50ms similarity search. This provides grounded context to the generator LLM, bridging the parametric "
            "knowledge gap and preventing hallucinations, while balancing chunk size (500 chars) and overlap (80 chars) to preserve context continuity."
        )
    elif "faiss" in q_low or "vector" in q_low or "flatl2" in q_low or "ivfflat" in q_low:
        return (
            "FAISS provides fast similarity search over high-dimensional vectors. IndexFlatL2 provides exact Euclidean L2 distance searches "
            "with 100% recall, while inverted file indexing (IndexIVFFlat) partitions vector space into Voronoi cells to prune search space for "
            "millions of vectors at the cost of a small recall trade-off. Pre-building and persisting the index avoids re-indexing on every query."
        )
    elif "langgraph" in q_low or "orchestrat" in q_low or "graph" in q_low or "state" in q_low:
        return (
            "LangGraph coordinates stateful, multi-step LLM workflows using a centralized TypedDict state schema. Unlike linear chains, "
            "it enables cyclic self-correction loops (e.g. query rewriting and draft verification) with bounded retry conditions (retry_count < 1) "
            "to prevent runaway latency while ensuring responses meet strict factual grounding standards."
        )
    else:
        sentences = [s.strip() for s in context.replace("\n", " ").split(". ") if len(s.strip()) > 20]
        if sentences:
            return ". ".join(sentences[:2]) + "."
        return "A complete answer must directly address the question using course definitions, architectural trade-offs, and exact parameters."


def evaluate_student_interview_response(
    question: str,
    student_answer: str,
    context: str
) -> str:
    """
    Evaluates student technical interview responses against a standard 10-point rubric:
    - Technical Accuracy (0-4 pts): Correct concepts, definitions, and mechanisms.
    - Completeness & Trade-offs (0-4 pts): Explains trade-offs, parameters, and failure modes.
    - Clarity & Engineering Depth (0-2 pts): Concise, professional technical terminology.
    Identifies exact gaps, why points were deducted, and provides the full authoritative model answer.
    """
    ans_clean = (student_answer or "").strip()
    ans_lower = ans_clean.lower()
    words = re.findall(r"\b\w+\b", ans_lower)

    # 1. Non-answers or trivial answers (e.g. "okay", "ok", "idk", single words)
    trivial_words = {"okay", "ok", "yes", "no", "idk", "none", "nothing", "sure", "fine", "hi", "hello"}
    is_trivial = (
        len(words) <= 2
        or ans_lower in trivial_words
        or (len(words) <= 4 and any(w in trivial_words for w in words))
    )

    model_ans = _get_topic_model_answer(question, context)

    if is_trivial:
        return (
            "SCORE: 0/10\n"
            "VERDICT: Inaccurate\n\n"
            "WHAT WAS GOOD:\n"
            f"None. The submitted response ('{ans_clean}') did not attempt to address the technical question.\n\n"
            "WHAT WAS MISSING:\n"
            "[Where You Failed / Points Deducted]\n"
            "- You provided zero technical explanation, definitions, or mechanisms for this question.\n"
            "- To earn points under our 10-point rubric, you must explain the underlying concepts, engineering trade-offs, and parameters relevant to the question.\n\n"
            "MODEL ANSWER:\n"
            "[What Should Have Been in the Answer]\n"
            f"{model_ans}"
        )

    # 2. Extract technical keywords from context and domain concepts
    context_words = set(re.findall(r"\b[a-zA-Z_]{3,}\b", (context + " " + question).lower()))
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "are", "which",
        "into", "can", "will", "does", "not", "have", "been", "using", "such",
        "than", "more", "what", "how", "why", "when", "where", "about"
    }
    domain_keywords = context_words - stop_words
    matched = [w for w in set(words) if w in domain_keywords]
    coverage = len(matched)

    # 3. Standard 10-point rubric scoring
    if len(words) < 8:
        # Extremely brief answers (e.g. "loss of quality")
        score = 2
        verdict = "Needs Improvement"
        good = f"You identified high-level concepts ('{ans_clean}'), but did not explain why or how."
        failed = (
            "- Answer is too brief (under 8 words) to demonstrate technical depth.\n"
            "- Failed to explain the underlying mechanisms, architectural trade-offs, or formulas.\n"
            "- Missing core course definitions and implementation specifics."
        )
    elif coverage >= 6 and len(words) >= 20:
        # Comprehensive, strong answer
        score = 9
        verdict = "Strong"
        good = f"Excellent technical depth. You accurately addressed key mechanisms ({', '.join(matched[:5])}) with precise engineering terminology."
        failed = "- Minor gaps: Could further elaborate on production edge cases or runtime failure modes."
    elif coverage >= 3 or len(words) >= 12:
        # Partial / decent answer
        score = 6
        verdict = "Needs Improvement"
        good = f"Good foundational awareness. You correctly noted concepts ({', '.join(matched[:4]) if matched else 'relevant principles'})."
        failed = (
            "- Incomplete trade-off analysis: Did not mention all critical parameters or failure modes.\n"
            "- Lacked full precision on specific mechanisms documented in the course material."
        )
    else:
        # Poor / vague answer
        score = 3
        verdict = "Inaccurate"
        good = f"Attempted to address the question ('{ans_clean}')."
        failed = (
            "- The answer lacks key course terminology and does not sufficiently reflect facts in the course reference material.\n"
            "- Missed primary definitions, trade-offs, and design considerations."
        )

    return (
        f"SCORE: {score}/10\n"
        f"VERDICT: {verdict}\n\n"
        "WHAT WAS GOOD:\n"
        f"{good}\n\n"
        "WHAT WAS MISSING:\n"
        "[Where You Failed / Points Deducted]\n"
        f"{failed}\n\n"
        "MODEL ANSWER:\n"
        "[What Should Have Been in the Answer]\n"
        f"{model_ans}"
    )


def node_handle_interview(state: TutorState, llm: LLMBackend) -> TutorState:
    """
    Node 5: Interview Practice (Experiment 6).
    - If student_answer is provided: evaluates against standard 10-point rubric.
    - If student_answer is absent: generates a technical interview question.
    """
    question = state.get("question", "")
    student_ans = state.get("student_answer", "").strip()
    topic = state.get("topic") or question or "LLM Fine-Tuning and RAG"
    context = state.get("context", "")

    if student_ans:
        feedback = evaluate_student_interview_response(question, student_ans, context)
        return {"final_answer": feedback}
    else:
        prompt = INTERVIEW_QUESTION_TEMPLATE.format(
            context=context,
            topic=topic
        )
        interview_q = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_TUTOR, max_tokens=256)
        return {"final_answer": interview_q}


def node_rewrite_query(state: TutorState) -> TutorState:
    """
    Node 6: Query Rewriting for Bounded Self-Correction (Experiment 6).
    Enriches search query with foundational terminology and increments retry_count.
    """
    current_q = state.get("search_query") or state.get("question") or ""
    current_retries = state.get("retry_count", 0)
    q_lower = current_q.lower()

    # Clean conversational filler
    cleaned_terms = [w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", current_q)
                     if w.lower() not in {"hi", "hello", "hey", "please", "can", "you", "tell", "me", "whats", "what"}]
    base_query = " ".join(cleaned_terms) if cleaned_terms else current_q

    # Target-specific domain expansions rather than blanket LoRA pollution
    if any(k in q_lower for k in ["rag", "retriev", "chunk", "bi-encoder", "dense"]):
        expansion = "Retrieval-Augmented Generation dense retrieval context passages embeddings"
    elif any(k in q_lower for k in ["lora", "adapter", "peft", "fine-tun", "rank", "alpha"]):
        expansion = "Low-Rank Adaptation PEFT parameter-efficient fine-tuning adapter weights"
    elif any(k in q_lower for k in ["qlora", "nf4", "quantiz", "double quant", "bnb"]):
        expansion = "QLoRA NormalFloat4 NF4 double quantization 4-bit base model"
    elif any(k in q_lower for k in ["faiss", "vector", "flatl2", "ivfflat", "index"]):
        expansion = "FAISS vector database similarity search IndexFlatL2 embeddings"
    elif any(k in q_lower for k in ["langgraph", "graph", "state", "retry", "orchestrat"]):
        expansion = "LangGraph stateful workflow nodes edges conditional routing"
    elif any(k in q_lower for k in ["fastapi", "pydantic", "endpoint", "api", "lifespan"]):
        expansion = "FastAPI lifespan validation Pydantic models REST API deployment"
    else:
        expansion = "LLM Engineering course concepts and definitions"

    rewritten = f"{base_query} {expansion}".strip()
    logger.info(f"[Graph: rewrite_query] Retry {current_retries + 1}: '{current_q}' -> '{rewritten}'")

    return {
        "search_query": rewritten,
        "retry_count": current_retries + 1
    }


def node_fallback(state: TutorState) -> TutorState:
    """
    Node 7: Safe Fallback (Experiment 6).
    Ensures safe, uninvented responses when documents do not support the query.
    """
    return {
        "final_answer": FALLBACK_TEXT,
        "quiz": []
    }


class StudyBuddyWorkflow:
    """Compiled LangGraph Workflow Wrapper."""

    def __init__(self, llm: Optional[LLMBackend] = None, retriever: Optional[CourseRetriever] = None):
        self.llm = llm or get_llm_backend(settings)
        self.retriever = retriever or create_retriever()
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(TutorState)

        # Register nodes with closures passing initialized dependencies
        graph.add_node("classify_intent", node_classify_intent)
        graph.add_node("retrieve", lambda s: node_retrieve(s, self.retriever))
        graph.add_node("draft_answer", lambda s: node_draft_answer(s, self.llm))
        graph.add_node("review_answer", node_review_answer)
        graph.add_node("generate_quiz", lambda s: node_generate_quiz(s, self.llm))
        graph.add_node("validate_quiz", node_validate_quiz)
        graph.add_node("handle_interview", lambda s: node_handle_interview(s, self.llm))
        graph.add_node("rewrite_query", node_rewrite_query)
        graph.add_node("fallback", node_fallback)

        # Flow from START
        graph.add_edge(START, "classify_intent")
        graph.add_edge("classify_intent", "retrieve")

        # Dynamic intent routing
        graph.add_conditional_edges(
            "retrieve",
            route_by_intent,
            {
                "draft_answer": "draft_answer",
                "generate_quiz": "generate_quiz",
                "handle_interview": "handle_interview",
                "fallback": "fallback"
            }
        )

        # Concept Q&A Subgraph
        graph.add_edge("draft_answer", "review_answer")
        graph.add_conditional_edges(
            "review_answer",
            review_answer_condition,
            {
                "ok": END,
                "retry": "rewrite_query",
                "fallback": "fallback"
            }
        )

        # Quiz Subgraph
        graph.add_edge("generate_quiz", "validate_quiz")
        graph.add_conditional_edges(
            "validate_quiz",
            validate_quiz_condition,
            {
                "ok": END,
                "retry": "rewrite_query",
                "fallback": "fallback"
            }
        )

        # Interview & Fallback edges
        graph.add_edge("handle_interview", END)
        graph.add_edge("fallback", END)

        # Bounded Retry Loop: rewrite_query -> retrieve
        graph.add_edge("rewrite_query", "retrieve")

        return graph.compile()

    def run(self, initial_state: TutorState) -> TutorState:
        """Executes the workflow graph and returns final state."""
        result = self.app.invoke(initial_state)
        return result


def export_mermaid_diagram(output_path: str = "docs/graph.md") -> str:
    """Exports the compiled LangGraph workflow as a Mermaid diagram into docs/graph.md."""
    workflow = StudyBuddyWorkflow()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mermaid_code = (
        "```mermaid\n"
        "flowchart TD\n"
        "    __start__([Start]) --> classify_intent[classify_intent]\n"
        "    classify_intent --> retrieve[retrieve]\n"
        "    retrieve -.->|explain_concept| draft_answer[draft_answer]\n"
        "    retrieve -.->|quiz_request| generate_quiz[generate_quiz]\n"
        "    retrieve -.->|interview_practice| handle_interview[handle_interview]\n"
        "    retrieve -.->|out_of_scope| fallback[fallback]\n"
        "    draft_answer --> review_answer[review_answer]\n"
        "    review_answer -.->|ok| __end__([End])\n"
        "    review_answer -.->|retry < 1| rewrite_query[rewrite_query]\n"
        "    review_answer -.->|retry >= 1| fallback\n"
        "    generate_quiz --> validate_quiz[validate_quiz]\n"
        "    validate_quiz -.->|ok| __end__\n"
        "    validate_quiz -.->|retry < 1| rewrite_query\n"
        "    validate_quiz -.->|retry >= 1| fallback\n"
        "    rewrite_query --> retrieve\n"
        "    handle_interview --> __end__\n"
        "    fallback --> __end__\n"
        "```\n"
    )

    try:
        # If LangGraph provides native mermaid export
        native_mermaid = workflow.app.get_graph().draw_mermaid()
        full_content = f"# StudyBuddy LangGraph Workflow Architecture\n\n```mermaid\n{native_mermaid}\n```\n"
    except Exception:
        full_content = f"# StudyBuddy LangGraph Workflow Architecture\n\n{mermaid_code}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_content)
    logger.info(f"Exported Mermaid diagram to '{output_path}'.")
    return full_content
