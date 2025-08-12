# rag.py
import faiss
import pickle
import os
import logging
import fitz  # PyMuPDF
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from semantic_chunker import SemanticChunker
from config import (
    EMBEDDING_MODEL_NAME, 
    INDEX_PATH, 
    PDF_DATA_PATH, 
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    CHUNKING_STRATEGY,
    MAX_CHUNK_TOKENS,
    EMBEDDING_BATCH_SIZE
)
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Manages document processing, chunking, embedding, and retrieval.
    """
    
    def __init__(self):
        self.embedding_model = None
        self.semantic_chunker = SemanticChunker()
        self.index = None
        self.texts = []
        self.metadata = []
        self._init_embedding_model()
        self._load_index()
    
    def _init_embedding_model(self):
        """Initializes the SentenceTransformer model."""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            self.embedding_model = None
    
    def _load_index(self):
        """Loads the FAISS index and associated metadata from disk."""
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
        """Extracts all text from a given PDF file."""
        try:
            with fitz.open(pdf_path) as doc:
                return "\n\n".join(page.get_text() for page in doc)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}", exc_info=True)
            return ""
    
    def chunk_document(self, text: str, source: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunks a document using the configured strategy."""
        chunks = []
        try:
            semantic_chunks = self.semantic_chunker.semantic_chunk(
                text, strategy=CHUNKING_STRATEGY, max_tokens=MAX_CHUNK_TOKENS
            )
            for i, (chunk_text, chunk_metadata) in enumerate(semantic_chunks):
                if len(chunk_text.strip()) < 50: continue
                metadata = {"source": source, "chunk_id": i, **chunk_metadata}
                chunks.append((chunk_text.strip(), metadata))
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}. Using fallback recursive chunking.")
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = text[i:i + CHUNK_SIZE].strip()
                if len(chunk_text) > 50:
                    metadata = {"source": source, "chunk_id": i // (CHUNK_SIZE - CHUNK_OVERLAP), "strategy": "fallback"}
                    chunks.append((chunk_text, metadata))
        return chunks
    
    def build_index(self, pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
        """Builds or rebuilds the FAISS index from PDF files in a directory."""
        if self.index is not None and not force_rebuild:
            logger.info("Index already exists. Use force_rebuild=True to rebuild.")
            return
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_directory}. Cannot build index.")
            return

        all_chunks, all_metadata = [], []
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(os.path.join(pdf_directory, pdf_file))
            if text:
                chunks = self.chunk_document(text, pdf_file)
                for chunk_text, metadata in chunks:
                    all_chunks.append(chunk_text)
                    all_metadata.append(metadata)
        
        if not all_chunks:
            logger.error("No text chunks were extracted from the PDFs.")
            return

        if not self.embedding_model:
            logger.error("Cannot build index because embedding model is not loaded.")
            return

        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            all_chunks, show_progress_bar=True, batch_size=EMBEDDING_BATCH_SIZE
        ).astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        self.texts = all_chunks
        self.metadata = all_metadata
        
        os.makedirs(INDEX_PATH, exist_ok=True)
        faiss.write_index(self.index, os.path.join(INDEX_PATH, "faiss.index"))
        with open(os.path.join(INDEX_PATH, "metadata.pkl"), "wb") as f:
            pickle.dump((self.texts, self.metadata), f)
        
        logger.info(f"Index built successfully with {self.index.ntotal} chunks.")
    
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Retrieves the top-k most relevant document chunks for a given query."""
        if self.index is None or not self.embedding_model:
            logger.warning("Cannot retrieve: RAG index or embedding model not available.")
            return []
        
        try:
            query_vector = self.embedding_model.encode([query]).astype('float32')
            distances, indices = self.index.search(query_vector, k)
            
            results = [
                {"text": self.texts[idx], "metadata": self.metadata[idx], "score": float(distances[0][i])}
                for i, idx in enumerate(indices[0]) if 0 <= idx < len(self.texts)
            ]
            logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            return []

# --- LAZY-LOADING AND PUBLIC API FUNCTIONS ---

_rag_system_instance = None
def get_rag_system():
    """Lazy-loads and returns a single instance of the RAGSystem."""
    global _rag_system_instance
    if _rag_system_instance is None:
        logger.info("Initializing RAGSystem for the first time...")
        _rag_system_instance = RAGSystem()
    return _rag_system_instance

def retrieve_and_format_context(query: str, k: int = TOP_K_RETRIEVAL) -> str:
    """
    **NEW FUNCTION**
    This is the main function for the conversational agent to call.
    It retrieves chunks and formats them into a single string for the LLM prompt.
    """
    rag_system = get_rag_system()
    results = rag_system.retrieve(query, k)
    if not results:
        return ""
    
    # Format the retrieved chunks into a single string block
    formatted_chunks = [f"Source: {res['metadata'].get('source', 'N/A')}, Chunk {res['metadata'].get('chunk_id', 'N/A')}\nContent: {res['text']}" for res in results]
    return "\n\n---\n\n".join(formatted_chunks)

def build_index(pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
    """Convenience function to trigger index building."""
    rag_system = get_rag_system()
    rag_system.build_index(pdf_directory, force_rebuild)
