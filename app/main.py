"""
FastAPI Application for StudyBuddy (Experiment 7).
Initializes all core components (Embeddings, FAISS Index, LLM Backend, LangGraph Workflow)
once at startup via lifespan context manager.
Enforces strict Pydantic validation, structured audit logging, latency tracking,
and sanitized error responses without leaking internal filepaths or stack traces.
"""

import time
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.ingestion import get_embedding_function, load_vectorstore
from app.llm_backends import LLMBackend, get_llm_backend
from app.logging_utils import log_request_event, logger
from app.retriever import CourseRetriever
from app.graph import StudyBuddyWorkflow
from app.tutor_rag import (
    ask_concept,
    generate_quiz_session,
    grade_quiz_session,
    practice_interview,
)


# ==========================================
# Lifespan Context Manager (Startup / Shutdown)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes heavy resources ONCE at startup:
    - Embedding model
    - Persisted FAISS vector index
    - LLM backend
    - LangGraph compiled workflow
    """
    logger.info("Initializing StudyBuddy components on server startup...")
    start_init = time.perf_counter()

    try:
        # Load embedding model
        app.state.embedding_fn = get_embedding_function(settings.EMBEDDING_MODEL_NAME)
        # Load persisted FAISS index
        app.state.vectorstore = load_vectorstore(settings.INDEX_PATH, app.state.embedding_fn)
        app.state.index_loaded = True
        logger.info("FAISS vector store successfully loaded into memory.")
    except Exception as e:
        logger.warning(f"Could not load persisted FAISS index: {e}. Running with empty retriever.")
        app.state.vectorstore = None
        app.state.index_loaded = False

    # Instantiate LLM backend
    app.state.llm_backend = get_llm_backend(settings)
    app.state.backend_name = settings.LLM_BACKEND

    # Initialize Retriever & LangGraph Workflow
    if app.state.vectorstore:
        app.state.retriever = CourseRetriever(app.state.vectorstore, k=settings.RETRIEVER_K)
    else:
        app.state.retriever = None

    app.state.workflow = StudyBuddyWorkflow(
        llm=app.state.llm_backend,
        retriever=app.state.retriever
    )

    elapsed_ms = (time.perf_counter() - start_init) * 1000
    logger.info(f"Startup initialization completed in {elapsed_ms:.1f}ms.")

    yield

    logger.info("StudyBuddy shutting down cleanly.")


app = FastAPI(
    title="StudyBuddy API",
    description="Course Tutor & Technical Interview Preparation Assistant for LLM Engineering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Middleware: Latency & Response-Time Header
# ==========================================
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Response-Time-ms"] = f"{latency_ms:.2f}"
    return response


# ==========================================
# Exception Handlers (Information Leaks Sanitized)
# ==========================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats validation errors clearly without leaking stack traces."""
    latency_ms = 0.0
    errors = exc.errors()
    msg = errors[0]["msg"] if errors else "Invalid request payload."
    log_request_event(
        endpoint=request.url.path,
        success=False,
        latency_ms=latency_ms,
        error_message=f"Validation error: {msg}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": f"Request validation failed: {msg}"}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global catch-all preventing stack traces, filepaths, or secrets in responses."""
    logger.exception(f"Unhandled exception during request to {request.url.path}: {exc}")
    log_request_event(
        endpoint=request.url.path,
        success=False,
        latency_ms=0.0,
        error_message="Internal server error"
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."}
    )


# ==========================================
# Pydantic Request & Response Schemas
# ==========================================
class AskRequest(BaseModel):
    question: str = Field(..., max_length=settings.MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace only.")
        return stripped


class AskResponse(BaseModel):
    answer: str
    intent: str
    sources: List[Dict[str, Any]]
    retries: int
    latency_ms: float


class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    num_questions: int = Field(3, ge=1, le=5)

    @field_validator("topic")
    @classmethod
    def validate_topic_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Topic cannot be empty or whitespace only.")
        return stripped


class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[Dict[str, Any]]
    latency_ms: float


class QuizGradeRequest(BaseModel):
    quiz_id: str = Field(..., min_length=5)
    answers: List[int] = Field(..., min_length=1)


class QuizGradeResponse(BaseModel):
    quiz_id: str
    score: int
    total: int
    percentage: float
    results: List[Dict[str, Any]]
    latency_ms: float


class InterviewRequest(BaseModel):
    topic: Optional[str] = Field(None, max_length=200)
    question: Optional[str] = Field(None, max_length=settings.MAX_QUESTION_LENGTH)
    student_answer: Optional[str] = Field(None, max_length=settings.MAX_QUESTION_LENGTH)

    @field_validator("question", "student_answer", "topic")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            return stripped if stripped else None
        return None


class InterviewResponse(BaseModel):
    feedback_or_question: str
    latency_ms: float


# ==========================================
# Endpoints
# ==========================================
@app.get("/health", summary="Health check and status")
async def health_check():
    """Returns application status, vector index state, and active LLM backend."""
    return {
        "status": "ok",
        "index_loaded": getattr(app.state, "index_loaded", False),
        "backend": getattr(app.state, "backend_name", "unknown")
    }


@app.post("/ask", response_model=AskResponse, summary="Ask a course concept question")
async def ask_question_endpoint(req: AskRequest):
    """
    Retrieves grounded context from course materials and returns an answer
    annotated with source citations, intent classification, and retries.
    """
    start_t = time.perf_counter()
    try:
        res = ask_concept(req.question, app.state.workflow)
        latency_ms = (time.perf_counter() - start_t) * 1000

        log_request_event(
            endpoint="/ask",
            success=True,
            latency_ms=latency_ms,
            intent=res.get("intent"),
            retries=res.get("retries", 0),
            user_query=req.question
        )

        return AskResponse(
            answer=res["answer"],
            intent=res["intent"],
            sources=res["sources"],
            retries=res["retries"],
            latency_ms=round(latency_ms, 2)
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_t) * 1000
        log_request_event(
            endpoint="/ask",
            success=False,
            latency_ms=latency_ms,
            user_query=req.question,
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing concept question."
        )


@app.post("/quiz", response_model=QuizResponse, summary="Generate a multiple-choice quiz")
async def generate_quiz_endpoint(req: QuizRequest):
    """
    Generates 1 to 5 multiple-choice questions from course documents.
    Answer keys are kept server-side and never returned to the client.
    """
    start_t = time.perf_counter()
    try:
        res = generate_quiz_session(req.topic, req.num_questions, app.state.workflow)
        latency_ms = (time.perf_counter() - start_t) * 1000

        log_request_event(
            endpoint="/quiz",
            success=True,
            latency_ms=latency_ms,
            intent="quiz_request",
            user_query=req.topic,
            extra={"num_questions": req.num_questions}
        )

        return QuizResponse(
            quiz_id=res["quiz_id"],
            questions=res["questions"],
            latency_ms=round(latency_ms, 2)
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_t) * 1000
        log_request_event(
            endpoint="/quiz",
            success=False,
            latency_ms=latency_ms,
            user_query=req.topic,
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating quiz."
        )


@app.post("/quiz/grade", response_model=QuizGradeResponse, summary="Grade submitted quiz answers")
async def grade_quiz_endpoint(req: QuizGradeRequest):
    """
    Deterministically grades student answers against server-stored answer keys.
    Returns score, per-question correctness, explanations, and source passages.
    """
    start_t = time.perf_counter()
    try:
        res = grade_quiz_session(req.quiz_id, req.answers)
        latency_ms = (time.perf_counter() - start_t) * 1000

        log_request_event(
            endpoint="/quiz/grade",
            success=True,
            latency_ms=latency_ms,
            intent="grade_quiz",
            extra={"quiz_id": req.quiz_id, "score": res["score"], "total": res["total"]}
        )

        return QuizGradeResponse(
            quiz_id=res["quiz_id"],
            score=res["score"],
            total=res["total"],
            percentage=res["percentage"],
            results=res["results"],
            latency_ms=round(latency_ms, 2)
        )
    except KeyError:
        latency_ms = (time.perf_counter() - start_t) * 1000
        log_request_event(
            endpoint="/quiz/grade",
            success=False,
            latency_ms=latency_ms,
            error_message="Unknown or expired quiz_id"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quiz ID '{req.quiz_id}' was not found or has expired."
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_t) * 1000
        log_request_event(
            endpoint="/quiz/grade",
            success=False,
            latency_ms=latency_ms,
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error grading quiz."
        )


@app.post("/interview", response_model=InterviewResponse, summary="Practice technical interviews")
async def interview_endpoint(req: InterviewRequest):
    """
    Technical interview practice:
    - If student_answer is provided: evaluates against rubric and returns structured feedback.
    - If question/student_answer is absent: generates a technical interview question for the topic.
    """
    start_t = time.perf_counter()
    try:
        res = practice_interview(
            topic=req.topic,
            question=req.question,
            student_answer=req.student_answer,
            workflow=app.state.workflow
        )
        latency_ms = (time.perf_counter() - start_t) * 1000

        log_request_event(
            endpoint="/interview",
            success=True,
            latency_ms=latency_ms,
            intent="interview_practice",
            user_query=req.question or req.topic or "interview_request"
        )

        return InterviewResponse(
            feedback_or_question=res["feedback_or_question"],
            latency_ms=round(latency_ms, 2)
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_t) * 1000
        log_request_event(
            endpoint="/interview",
            success=False,
            latency_ms=latency_ms,
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during interview practice session."
        )


# ==========================================
# Frontend Static Route (Single Page HTML)
# ==========================================
@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = "frontend/index.html"
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "StudyBuddy API is running. Visit /docs for OpenAPI documentation."}
