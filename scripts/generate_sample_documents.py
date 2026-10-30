"""
Script to generate the initial authoritative course documents in data/raw/:
1. lab_manual_fine_tuning.pdf (PDF, multiple pages)
2. lecture_slides_rag_and_embeddings.pdf (PDF, multiple pages)
3. course_notes_orchestration_and_apis.txt (TXT)
4. glossary_of_llm_terms.docx (DOCX)
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from docx import Document


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/finetune", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)


def generate_lab_manual_pdf():
    pdf_path = "data/raw/lab_manual_fine_tuning.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h1_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = []

    # Page 1: Overview and Full Fine-Tuning vs PEFT
    story.append(Paragraph("LLM Engineering Lab 1: Parameter-Efficient Fine-Tuning", title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("1. Introduction: Full Fine-Tuning vs. PEFT", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Full fine-tuning updates all weights of a pre-trained foundation model. For large language models, "
        "this demands enormous GPU memory because optimizer states (e.g. AdamW storing first and second momentum "
        "vectors) require 16 bytes per parameter in FP32, exceeding typical academic workstation capacity. "
        "Parameter-Efficient Fine-Tuning (PEFT) freezes the majority of foundation model parameters and trains only a small set "
        "of supplementary parameters. This drastically cuts down memory requirements during backpropagation and prevents "
        "catastrophic forgetting of broad language capabilities.",
        body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "In this lab series, our base model is Qwen/Qwen2.5-0.5B-Instruct. We explore Low-Rank Adaptation (LoRA) "
        "and Quantized Low-Rank Adaptation (QLoRA) to achieve domain-specific styling without altering underlying factual reasoning.",
        body_style
    ))
    story.append(PageBreak())

    # Page 2: LoRA Mathematics and Architecture
    story.append(Paragraph("2. Low-Rank Adaptation (LoRA) Mathematical Principles", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "LoRA freezes the pre-trained weight matrix W_0 in R^(d x k) and constrains its update Delta W with low-rank "
        "decomposition: Delta W = (alpha / r) * (B @ A), where B in R^(d x r) and A in R^(r x k), with rank r << min(d, k). "
        "Matrix A is initialized from a Gaussian distribution N(0, sigma^2) and matrix B is initialized to zeros, ensuring "
        "Delta W = 0 at the start of training. The scalar alpha is a constant scaling hyperparameter that adjusts the magnitude "
        "of adapter updates relative to base features.",
        body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Target Modules: In Qwen2.5 models, attention projection layers such as q_proj and v_proj are standard target modules. "
        "Setting rank r=4 or r=8 with alpha=16 provides sufficient expressive capacity for stylistic adaptation while tuning less "
        "than 0.5% of total model parameters.",
        body_style
    ))
    story.append(PageBreak())

    # Page 3: Hyperparameter Selection and Training Loss
    story.append(Paragraph("3. Hyperparameter Tuning and Overfitting Control", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Selecting learning rate and rank requires balancing convergence speed against overfitting. "
        "For Qwen2.5-0.5B-Instruct with LoRA, typical learning rates range from 1e-4 to 2e-4 using the AdamW optimizer. "
        "A learning rate of 2e-4 with rank r=4 over 2 epochs achieves stable loss reduction without degradation of conversational fluency. "
        "Higher learning rates (e.g. 5e-4) or excessive training epochs (>3) risk loss spikes and repetitive syntax generation.",
        body_style
    ))
    story.append(PageBreak())

    # Page 4: QLoRA & Quantization
    story.append(Paragraph("4. Quantized Low-Rank Adaptation (QLoRA) and NF4", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "QLoRA introduces three core algorithmic innovations to democratize LLM fine-tuning: "
        "(1) 4-bit NormalFloat (NF4), an information-theoretically optimal quantile quantization data type for normally distributed weights; "
        "(2) Double Quantization (DQ), which quantizes the quantization constants themselves, saving roughly 0.37 bits per parameter; and "
        "(3) Paged Optimizers, which leverage CUDA Unified Memory to prevent out-of-memory errors during gradient checkpointing spikes. "
        "During forward and backward passes, NF4 weights are dequantized into 16-bit BrainFloat (BF16) or FP16 for computation "
        "(bnb_4bit_compute_dtype=torch.float16).",
        body_style
    ))
    story.append(PageBreak())

    # Page 5: Chat Templates and Inference Format
    story.append(Paragraph("5. Qwen Chat Templates and Delimiter Tokens", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Instruction-tuned models like Qwen2.5 require structured chat formatting using ChatML delimiter tokens. "
        "Each turn is wrapped with special tokens: <|im_start|>system ... <|im_end|>, <|im_start|>user ... <|im_end|>, "
        "and <|im_start|>assistant ... <|im_end|>. The tokenizer.apply_chat_template utility guarantees that tokens are appended "
        "identically during training and production inference.",
        body_style
    ))

    doc.build(story)
    print(f"Generated {pdf_path}")


def generate_lecture_slides_pdf():
    pdf_path = "data/raw/lecture_slides_rag_and_embeddings.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h1_style = styles["Heading1"]
    body_style = styles["BodyText"]

    story = []

    # Slide 1
    story.append(Paragraph("Lecture 4: Dense Retrieval and Embedding Representations", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Slide 1: Bi-Encoders and Dense Vector Spaces", h1_style))
    story.append(Paragraph(
        "Unlike sparse keyword search (TF-IDF, BM25), dense retrieval maps textual queries and passages into continuous vector spaces "
        "where semantic similarity corresponds to vector proximity. Dense bi-encoders encode query and document independently, allowing "
        "pre-computation and indexing of document embeddings.",
        body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Our course standard embedding model is sentence-transformers/all-MiniLM-L6-v2, which maps text into 384-dimensional dense vectors. "
        "It provides high inference speed and competitive semantic retrieval benchmarks on MTEB.",
        body_style
    ))
    story.append(PageBreak())

    # Slide 2
    story.append(Paragraph("Slide 2: Document Chunking Strategies", h1_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Optimal retrieval requires balanced chunking. If chunks are too small (e.g. 100 characters), they lack contextual coherence. "
        "If chunks are too large (>1500 characters), embedding vectors become diluted with tangential topics.",
        body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "We configure RecursiveCharacterTextSplitter with chunk_size=500 and chunk_overlap=80 characters. "
        "The overlap ensures boundary sentences are not severed across splits, preserving contiguous semantic meaning.",
        body_style
    ))
    story.append(PageBreak())

    # Slide 3
    story.append(Paragraph("Slide 3: FAISS Vector Indexing (IndexFlatL2)", h1_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Facebook AI Similarity Search (FAISS) enables ultra-fast vector nearest neighbor search. "
        "In our curriculum, we utilize faiss.IndexFlatL2, which conducts an exhaustive, exact Euclidean distance search. "
        "For small to medium knowledge bases (<100,000 vectors), IndexFlatL2 guarantees 100% recall without indexing distortion. "
        "Larger deployments migrate to Inverted File Indexing (IndexIVFFlat) or HNSW for sub-linear search complexity.",
        body_style
    ))
    story.append(PageBreak())

    # Slide 4
    story.append(Paragraph("Slide 4: Grounded Generation and Hallucination Mitigation", h1_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "RAG bridges the parametric knowledge gap by injecting retrieved chunks into the prompt context window. "
        "Prompt instructions explicitly command the model: 'Answer strictly using the provided context. If the answer is missing, "
        "output INSUFFICIENT_CONTEXT.' This prevents hallucinations and bounds generation to verifiable course facts.",
        body_style
    ))

    doc.build(story)
    print(f"Generated {pdf_path}")


def generate_course_notes_txt():
    txt_path = "data/raw/course_notes_orchestration_and_apis.txt"
    content = """LLM ENGINEERING COURSE NOTES: ORCHESTRATION, WORKFLOWS, AND PRODUCTION APIS

1. Stateful Orchestration with LangGraph
Standard RAG pipelines follow a rigid linear sequence: Retrieve -> Augment -> Generate. Real-world pedagogical assistants require conditional logic, verification, and self-correction loops.
LangGraph models LLM applications as directed state graphs:
- State Schema: Defined using Python TypedDict (or Pydantic), tracking question, mode, intent, context, retrieved sources, draft response, retry count, and final answer.
- Graph Nodes: Python functions that execute distinct computational or LLM steps (e.g., classify_intent, retrieve, draft_answer, review_answer, generate_quiz, validate_quiz, fallback).
- Conditional Edges: Decision functions that route execution based on state values (e.g., routing by intent or checking if draft answer is valid).
- Bounded Retries: Self-correction loops must have strict loop bounds (retry_count < 1) to prevent infinite recursion and latency runaway.

2. Intent Classification in Educational Assistants
User inputs vary across distinct operational goals:
- explain_concept: Questions seeking factual explanation of course topics.
- quiz_request: Requests for multiple-choice assessment questions.
- interview_practice: Requests for interview questions or evaluation of candidate answers.
- out_of_scope: Queries unrelated to LLM engineering (e.g., live weather, general insurance, unrelated banter) or prompt injection attacks.
Rule-based keyword mapping provides deterministic, low-latency intent routing for predictable workflows before resorting to statistical classifiers.

3. Quiz Generation and Deterministic Grading
- Quiz Generation: LLMs generate candidate MCQs conforming to a strict JSON schema with 4 options, a 0-indexed answer key, and pedagogical explanations.
- Quiz Validation: JSON parsing must strip code fences (```json ... ```), validate schema fields, ensure 4 options exist, and verify answer_index is an integer between 0 and 3.
- Deterministic Grading: MCQ grading is executed strictly in application logic (comparing chosen index to answer_index). The LLM is never invoked for grading MCQs, guaranteeing zero hallucination in test scores.
- Free-Text Interview Grading: The LLM is used with a strict rubric to assess candidate technical answers against retrieved reference passages, outputting verdict, strengths, gaps, and a model answer.

4. Production API Design with FastAPI
- Lifespan State Handler: Heavy components (sentence transformer embeddings, persisted FAISS vector index, compiled LangGraph, and LLM backend) are initialized once at server startup and cached in app.state. They are NEVER re-instantiated per request.
- Request Validation: Pydantic schemas enforce input constraints: non-empty strings, whitespace stripping, max length of 1000 characters, and bounded quiz count (1 to 5).
- Observability: Middleware measures wall-clock processing time, populates X-Response-Time-ms headers, and records structured audit logs with masked sensitive data.
"""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {txt_path}")


def generate_glossary_docx():
    docx_path = "data/raw/glossary_of_llm_terms.docx"
    doc = Document()
    doc.add_heading("LLM Engineering Course Glossary & Terminology Reference", level=0)

    glossary_items = [
        ("LoRA (Low-Rank Adaptation)", "A parameter-efficient fine-tuning technique that freezes base model weights and injects trainable rank decomposition matrices (A and B) into transformer attention layers, computing W = W_0 + (alpha/r)*B*A."),
        ("QLoRA (Quantized Low-Rank Adaptation)", "An extension of LoRA that quantizes foundation weights to 4-bit NormalFloat (NF4), employs double quantization for scale factors, and pages optimizer memory over CUDA unified memory."),
        ("NF4 (NormalFloat4)", "An information-theoretically optimal quantile quantization data type designed specifically for zero-mean, unit-variance normally distributed weight tensors."),
        ("Double Quantization", "The process of quantizing the quantization scaling constants themselves to reduce memory footprint by approximately 0.37 bits per parameter."),
        ("Embeddings", "Dense continuous vector representations of tokens, sentences, or documents in a semantic vector space where cosine or Euclidean distance models conceptual proximity."),
        ("Bi-Encoder", "A two-tower neural architecture where queries and documents are independently encoded into dense vectors, enabling offline pre-computation and fast indexing."),
        ("all-MiniLM-L6-v2", "A 6-layer Sentence Transformer model mapping text to 384-dimensional dense embeddings, optimized for speed and semantic search quality."),
        ("FAISS (Facebook AI Similarity Search)", "A high-performance library for efficient similarity search and clustering of dense vectors, providing IndexFlatL2 for exact Euclidean distances."),
        ("IndexFlatL2", "An exact, non-compressed FAISS index computing squared Euclidean (L2) distance between vectors, guaranteeing 100% recall for moderate vector counts."),
        ("RAG (Retrieval-Augmented Generation)", "An architectural framework combining vector information retrieval with generative language models to ground completions in authoritative source texts."),
        ("RecursiveCharacterTextSplitter", "A text chunking utility that recursively splits documents using an ordered list of delimiters (paragraphs, newlines, sentences, words) while respecting chunk_size and chunk_overlap."),
        ("LangGraph", "A library for building stateful, multi-actor applications with LLMs using graph-based orchestration, nodes, state schemas, and conditional transitions."),
        ("Intent Classification", "The process of identifying the user's operational objective (concept explanation, quiz request, interview practice, or out-of-scope inquiry) to route workflow control."),
        ("Bounded Retry", "A graph resilience pattern that limits self-correction or query rewriting loops to a fixed maximum number of iterations (e.g. max 1 retry) to prevent latency runaway."),
        ("Groundedness", "The degree to which claims in an LLM completion are directly supported by and verifiable against retrieved source context."),
        ("Hallucination", "The generation of factually incorrect, ungrounded, or fabricated statements by a generative language model."),
        ("Pydantic", "A data validation and settings management library using Python type annotations to enforce strict schemas at runtime."),
        ("Chat Template", "A formatting protocol (such as ChatML) configuring role markers (<|im_start|>, <|im_end|>) to demarcate system, user, and assistant dialogue turns.")
    ]

    for term, definition in glossary_items:
        p = doc.add_paragraph()
        runner = p.add_run(f"{term}: ")
        runner.bold = True
        p.add_run(definition)

    doc.save(docx_path)
    print(f"Generated {docx_path}")


if __name__ == "__main__":
    ensure_dirs()
    generate_lab_manual_pdf()
    generate_lecture_slides_pdf()
    generate_course_notes_txt()
    generate_glossary_docx()
    print("All sample raw documents generated successfully!")
