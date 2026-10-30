"""
Retriever Module for StudyBuddy.
Provides similarity search over persisted FAISS vector index with metadata extraction.
Formats context passages and provides structured source citations (file and page).
"""

from typing import Dict, List, Optional, Tuple, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.ingestion import load_vectorstore, get_embedding_function
from app.logging_utils import logger


class CourseRetriever:
    """Retriever wrapper querying the persisted FAISS vector store."""

    def __init__(self, vectorstore: Optional[FAISS] = None, k: int = settings.RETRIEVER_K):
        self.k = k
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            embedding_fn = get_embedding_function(settings.EMBEDDING_MODEL_NAME)
            self.vectorstore = load_vectorstore(settings.INDEX_PATH, embedding_fn)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Performs similarity search returning top-k Document chunks."""
        top_k = k if k is not None else self.k
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return docs

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Performs similarity search returning documents with L2 distance scores."""
        top_k = k if k is not None else self.k
        return self.vectorstore.similarity_search_with_score(query, k=top_k)

    def format_context_and_sources(
        self,
        docs: List[Document]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Formats retrieved documents into a coherent context prompt block
        and extracts a clean, deduplicated list of source citations.
        """
        context_blocks: List[str] = []
        sources: List[Dict[str, Any]] = []
        seen_citations = set()

        for idx, doc in enumerate(docs, start=1):
            src_file = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", 1)
            topic = doc.metadata.get("topic", "")

            citation_key = (src_file, page_num)
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                sources.append({
                    "source": src_file,
                    "page": page_num,
                    "topic": topic
                })

            passage_header = f"[Source {idx}: {src_file} (Page {page_num})]"
            context_blocks.append(f"{passage_header}\n{doc.page_content.strip()}")

        context_str = "\n\n".join(context_blocks)
        return context_str, sources


def create_retriever(vectorstore: Optional[FAISS] = None, k: int = settings.RETRIEVER_K) -> CourseRetriever:
    """Helper factory for CourseRetriever."""
    return CourseRetriever(vectorstore=vectorstore, k=k)
