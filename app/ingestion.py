"""
Document Ingestion and Vector Store Pipeline (Experiment 5).
Loads multi-format documents (PDF, TXT, DOCX), splits into contiguous chunks,
extracts rich metadata, generates dense embeddings, and persists a FAISS vector index.

Usage:
    python -m app.ingestion --rebuild
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.config import Settings, settings
from app.logging_utils import logger


TOPIC_MAPPING = {
    "lab_manual": "fine_tuning_and_peft",
    "lecture_slides": "rag_and_embeddings",
    "course_notes": "orchestration_and_apis",
    "glossary": "course_glossary_and_definitions",
}


def derive_topic_from_filename(filename: str) -> str:
    """Derives a semantic topic identifier from document filename."""
    lower_name = filename.lower()
    for key, topic in TOPIC_MAPPING.items():
        if key in lower_name:
            return topic
    return "general_llm_engineering"


def load_raw_documents(raw_dir: str) -> List[Document]:
    """
    Loads documents from raw_dir across supported formats: PDF, TXT, DOCX.
    Preserves source and page metadata. Emits warnings for empty documents.
    """
    path = Path(raw_dir)
    if not path.exists():
        logger.warning(f"Raw document directory '{raw_dir}' does not exist.")
        return []

    documents: List[Document] = []

    for file_path in sorted(path.iterdir()):
        if file_path.is_dir():
            continue

        ext = file_path.suffix.lower()
        file_name = file_path.name
        topic = derive_topic_from_filename(file_name)

        loaded_docs: List[Document] = []
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                raw_pages = loader.load()
                for p in raw_pages:
                    text_content = p.page_content.strip()
                    if not text_content:
                        logger.warning(f"Empty text encountered in PDF: '{file_name}' on page {p.metadata.get('page', 0)}")
                        continue
                    # Normalize page number to 1-indexed
                    page_num = p.metadata.get("page", 0) + 1
                    p.metadata["source"] = file_name
                    p.metadata["page"] = page_num
                    p.metadata["topic"] = topic
                    loaded_docs.append(p)

            elif ext == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                raw_docs = loader.load()
                for d in raw_docs:
                    if not d.page_content.strip():
                        logger.warning(f"Empty text encountered in TXT file: '{file_name}'")
                        continue
                    d.metadata["source"] = file_name
                    d.metadata["page"] = 1
                    d.metadata["topic"] = topic
                    loaded_docs.append(d)

            elif ext == ".docx":
                loader = Docx2txtLoader(str(file_path))
                raw_docs = loader.load()
                for d in raw_docs:
                    if not d.page_content.strip():
                        logger.warning(f"Empty text encountered in DOCX file: '{file_name}'")
                        continue
                    d.metadata["source"] = file_name
                    d.metadata["page"] = 1
                    d.metadata["topic"] = topic
                    loaded_docs.append(d)

            else:
                logger.debug(f"Skipping unsupported file format: {file_name}")
                continue

            logger.info(f"Loaded {len(loaded_docs)} sections from '{file_name}'")
            documents.extend(loaded_docs)

        except Exception as e:
            logger.error(f"Error loading document '{file_name}': {e}")

    return documents


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Wrapper conforming to LangChain Embeddings interface for sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
        return embedding.tolist()


def get_embedding_function(model_name: str = settings.EMBEDDING_MODEL_NAME):
    """Returns the shared embedding function."""
    return LocalSentenceTransformerEmbeddings(model_name)


def split_documents(
    documents: List[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[Document]:
    """Splits documents using RecursiveCharacterTextSplitter with metadata preservation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document sections into {len(chunks)} text chunks.")
    return chunks


def build_and_persist_index(
    raw_dir: str = settings.RAW_DATA_DIR,
    index_path: str = settings.INDEX_PATH,
    processed_dir: str = settings.PROCESSED_DATA_DIR,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> FAISS:
    """
    Ingests all raw documents, chunks them, computes embeddings, and persists FAISS index.
    Also exports a chunk dump JSON to processed_dir for manual inspection.
    """
    logger.info(f"Starting document ingestion from '{raw_dir}'...")
    raw_docs = load_raw_documents(raw_dir)
    if not raw_docs:
        raise ValueError(f"No valid documents found in '{raw_dir}'. Ingestion aborted.")

    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Export chunks for audit and debugging inspection
    os.makedirs(processed_dir, exist_ok=True)
    chunk_dump_file = Path(processed_dir) / "chunks_dump.json"
    dump_data = [
        {
            "chunk_id": idx,
            "source": c.metadata.get("source"),
            "page": c.metadata.get("page"),
            "topic": c.metadata.get("topic"),
            "char_length": len(c.page_content),
            "preview": c.page_content[:150] + "..." if len(c.page_content) > 150 else c.page_content
        }
        for idx, c in enumerate(chunks)
    ]
    with open(chunk_dump_file, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=2)
    logger.info(f"Exported chunk inspection dump to '{chunk_dump_file}'")

    # Compute source statistics
    source_counts: Dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\n" + "=" * 50)
    print("INGESTION CHUNK COUNTS PER SOURCE DOCUMENT")
    print("=" * 50)
    for src, count in sorted(source_counts.items()):
        print(f"  • {src:<42} : {count:>3} chunks")
    print(f"  Total chunks indexed: {len(chunks)}")
    print("=" * 50 + "\n")

    embedding_fn = get_embedding_function(settings.EMBEDDING_MODEL_NAME)
    logger.info(f"Indexing {len(chunks)} chunks into FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embedding_fn)

    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    logger.info(f"FAISS index successfully persisted to '{index_path}'.")

    return vectorstore


def load_vectorstore(
    index_path: str = settings.INDEX_PATH,
    embedding_fn: Optional[LocalSentenceTransformerEmbeddings] = None
) -> FAISS:
    """Loads existing persisted FAISS vector store from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. Run 'python -m app.ingestion --rebuild' first."
        )
    emb_fn = embedding_fn or get_embedding_function(settings.EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(
        index_path,
        emb_fn,
        allow_dangerous_deserialization=True
    )
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StudyBuddy Document Ingestion & FAISS Indexer")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index from scratch")
    args = parser.parse_args()

    build_and_persist_index()
