# rag.py
import os
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional

import faiss
import fitz  # PyMuPDF
import numpy as np

from embeddings import get_embedding_model
from semantic_chunker import SemanticChunker
from context_formatter import context_formatter
from config import (
    INDEX_PATH, PDF_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
    CHUNKING_STRATEGY, MAX_CHUNK_TOKENS, EMBEDDING_BATCH_SIZE
)

logger = logging.getLogger(__name__)

class RAGSystem:
    """Manages document processing, chunking, embedding, and retrieval."""
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.semantic_chunker = SemanticChunker()
        self.index = None
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self):
        index_file = os.path.join(INDEX_PATH, "faiss.index")
        metadata_file = os.path.join(INDEX_PATH, "metadata.pkl")
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, "rb") as f:
                    self.texts, self.metadata = pickle.load(f)
                logger.info(f"Loaded RAG index with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Failed to load index files: {e}", exc_info=True)
        else:
            logger.warning("No existing index found. Run build_index() to create one.")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                return "\n\n".join(page.get_text() for page in doc)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}", exc_info=True)
            return ""

    def chunk_document(self, text: str, source: str) -> List[Tuple[str, Dict[str, Any]]]:
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        try:
            semantic_chunks = self.semantic_chunker.semantic_chunk(
                text, strategy=CHUNKING_STRATEGY, max_tokens=MAX_CHUNK_TOKENS
            )
            for i, (chunk_text, chunk_metadata) in enumerate(semantic_chunks):
                if len(chunk_text.strip()) < 50:
                    continue
                metadata = {"source": source, "chunk_id": i, **chunk_metadata}
                chunks.append((chunk_text.strip(), metadata))
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}. Using fallback recursive chunking.")
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for i in range(0, len(text), step):
                chunk_text = text[i:i + CHUNK_SIZE].strip()
                if len(chunk_text) > 50:
                    metadata = {"source": source, "chunk_id": i // step, "strategy": "fallback"}
                    chunks.append((chunk_text, metadata))
        return chunks

    def build_index(self, pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
        if self.index is not None and not force_rebuild:
            logger.info("Index already exists. Use force_rebuild=True to rebuild.")
            return

        if self.embedding_model is None:
            logger.error("Cannot build index because embedding model failed to load.")
            return

        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')] if os.path.isdir(pdf_directory) else []
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_directory}. Cannot build index.")
            return

        all_chunks, all_metadata = [], []
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(os.path.join(pdf_directory, pdf_file))
            if text:
                for chunk_text, metadata in self.chunk_document(text, pdf_file):
                    all_chunks.append(chunk_text)
                    all_metadata.append(metadata)

        if not all_chunks:
            logger.error("No text chunks were extracted from the PDFs.")
            return

        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            all_chunks, show_progress_bar=True, batch_size=EMBEDDING_BATCH_SIZE
        ).astype('float32')

        # Use cosine similarity via L2 normalization + IndexFlatIP
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.texts = all_chunks
        self.metadata = all_metadata

        os.makedirs(INDEX_PATH, exist_ok=True)
        faiss.write_index(self.index, os.path.join(INDEX_PATH, "faiss.index"))
        with open(os.path.join(INDEX_PATH, "metadata.pkl"), "wb") as f:
            pickle.dump((self.texts, self.metadata), f)

        logger.info(f"Index built successfully with {self.index.ntotal} chunks.")

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        if self.index is None or self.embedding_model is None:
            logger.warning("Cannot retrieve: RAG index or embedding model not available.")
            return []
        try:
            query_vec = self.embedding_model.encode([query]).astype('float32')
            faiss.normalize_L2(query_vec)
            distances, indices = self.index.search(query_vec, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.texts):
                    results.append({
                        "text": self.texts[idx],
                        "metadata": self.metadata[idx],
                        "score": float(distances[0][i])
                    })
            logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            return []

# --- Public API ---

_rag_system_instance: Optional[RAGSystem] = None
def get_rag_system() -> RAGSystem:
    global _rag_system_instance
    if _rag_system_instance is None:
        logger.info("Initializing RAGSystem for the first time...")
        _rag_system_instance = RAGSystem()
    return _rag_system_instance

def retrieve_and_format_context(query: str, k: int = TOP_K_RETRIEVAL) -> str:
    rag_system = get_rag_system()
    results = rag_system.retrieve(query, k)
    if not results:
        return ""

    # Assemble "Source + Content" chunks, then let the formatter deduplicate/trim
    chunks = [
        f"Source: {res['metadata'].get('source','N/A')}, "
        f"Chunk {res['metadata'].get('chunk_id','N/A')}\n"
        f"{res['text']}".strip()
        for res in results
    ]
    formatted = context_formatter.format_enhanced_context(chunks, query)
    return formatted

def build_index(pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
    rag_system = get_rag_system()
    rag_system.build_index(pdf_directory, force_rebuild)
