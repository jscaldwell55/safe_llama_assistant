# rag.py - Optimized RAG for 100-200 page documents with Apple Silicon fixes

import os
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional
import faiss
import fitz  # PyMuPDF
import numpy as np
import torch

from embeddings import get_embedding_model
from semantic_chunker import SemanticChunker
from context_formatter import format_retrieved_context
from config import (
    INDEX_PATH, PDF_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
    CHUNKING_STRATEGY, MAX_CHUNK_TOKENS, EMBEDDING_BATCH_SIZE, MAX_CONTEXT_LENGTH
)

logger = logging.getLogger(__name__)

class OptimizedRAGSystem:
    """RAG system optimized for 100-200 page documents"""
    
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.semantic_chunker = SemanticChunker()
        self.index = None
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.doc_count = 0
        self.total_chunks = 0
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index if available - FIXED for compatibility"""
        index_file = os.path.join(INDEX_PATH, "faiss.index")
        metadata_file = os.path.join(INDEX_PATH, "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, "rb") as f:
                    saved_data = pickle.load(f)
                    
                    # Handle both old tuple format and new dict format
                    if isinstance(saved_data, tuple):
                        # Old format: (texts, metadata)
                        logger.info("Loading old format index (tuple)")
                        self.texts, self.metadata = saved_data
                        self.doc_count = 0  # Unknown in old format
                    elif isinstance(saved_data, dict):
                        # New format: {"texts": ..., "metadata": ..., "doc_count": ...}
                        logger.info("Loading new format index (dict)")
                        self.texts = saved_data["texts"]
                        self.metadata = saved_data["metadata"]
                        self.doc_count = saved_data.get("doc_count", 0)
                    else:
                        raise ValueError(f"Unknown saved data format: {type(saved_data)}")
                    
                    self.total_chunks = len(self.texts)
                
                logger.info(f"Loaded RAG index: {self.doc_count} documents, {self.total_chunks} chunks")
            except Exception as e:
                logger.error(f"Failed to load index: {e}", exc_info=True)
                self.index = None
                self.texts = []
                self.metadata = []
                self.doc_count = 0
                self.total_chunks = 0
        else:
            logger.warning("No existing index found. Run build_index() to create one.")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with better formatting preservation"""
        try:
            logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
            with fitz.open(pdf_path) as doc:
                pages_text = []
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text()
                    if text.strip():
                        pages_text.append(f"[Page {page_num}]\n{text}")
                
                full_text = "\n\n".join(pages_text)
                logger.info(f"Extracted {len(full_text)} characters from {len(doc)} pages")
                return full_text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}", exc_info=True)
            return ""
    
    def chunk_document(self, text: str, source: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Create optimized chunks for retrieval"""
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        
        try:
            # Use semantic chunking for better context preservation
            semantic_chunks = self.semantic_chunker.semantic_chunk(
                text, 
                strategy=CHUNKING_STRATEGY, 
                max_tokens=MAX_CHUNK_TOKENS
            )
            
            for i, (chunk_text, chunk_metadata) in enumerate(semantic_chunks):
                # Skip very small chunks
                if len(chunk_text.strip()) < 100:
                    continue
                
                # Clean and normalize chunk text
                chunk_text = self._clean_chunk_text(chunk_text)
                
                # Create metadata
                metadata = {
                    "source": source,
                    "chunk_id": i,
                    "chunk_size": len(chunk_text),
                    **chunk_metadata
                }
                
                chunks.append((chunk_text, metadata))
                
            logger.info(f"Created {len(chunks)} chunks from {source}")
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}. Using fallback.")
            # Fallback to simple overlapping chunks
            chunks = self._fallback_chunking(text, source)
        
        return chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean chunk text for better retrieval"""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove page markers if at the beginning
        text = re.sub(r'^\[Page \d+\]\s*', '', text)
        
        return text.strip()
    
    def _fallback_chunking(self, text: str, source: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Simple overlapping chunk strategy as fallback"""
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + CHUNK_SIZE].strip()
            if len(chunk_text) > 100:
                metadata = {
                    "source": source,
                    "chunk_id": i // step,
                    "strategy": "fallback",
                    "chunk_size": len(chunk_text)
                }
                chunks.append((chunk_text, metadata))
        
        return chunks
    
    def _generate_embeddings_safe(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings with multiple fallback strategies"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Try progressively smaller batch sizes
        batch_sizes = [4, 2, 1]
        
        for batch_size in batch_sizes:
            try:
                logger.info(f"Attempting with batch_size={batch_size}")
                
                # Disable multiprocessing which can cause segfaults
                torch.set_num_threads(1)
                
                embeddings = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch,
                        show_progress_bar=False,  # Disable progress bar for batch processing
                        convert_to_tensor=False,
                        device='cpu',
                        normalize_embeddings=False
                    )
                    embeddings.append(batch_embeddings)
                    
                    # Log progress
                    if i % 20 == 0:
                        logger.info(f"Processed {i}/{len(chunks)} chunks")
                
                # Concatenate all embeddings
                embeddings = np.vstack(embeddings).astype('float32')
                logger.info(f"Successfully generated embeddings with batch_size={batch_size}")
                return embeddings
                
            except Exception as e:
                logger.error(f"Failed with batch_size={batch_size}: {e}")
                if batch_size == 1:
                    # If even batch_size=1 fails, we have a bigger problem
                    raise
                continue
        
        raise RuntimeError("Could not generate embeddings with any batch size")
    
    def build_index(self, pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
        """Build optimized FAISS index for 100-200 pages"""
        if self.index is not None and not force_rebuild:
            logger.info("Index already exists. Use force_rebuild=True to rebuild.")
            return
        
        if self.embedding_model is None:
            logger.error("Cannot build index: embedding model not loaded.")
            return
        
        # Find PDF files
        pdf_files = []
        if os.path.isdir(pdf_directory):
            pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
            logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory}")
        
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_directory}")
            return
        
        # Process each PDF
        all_chunks = []
        all_metadata = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            logger.info(f"Processing: {pdf_file}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_file}")
                continue
            
            # Create chunks
            chunks = self.chunk_document(text, pdf_file)
            for chunk_text, metadata in chunks:
                all_chunks.append(chunk_text)
                all_metadata.append(metadata)
        
        if not all_chunks:
            logger.error("No chunks created from PDFs")
            return
        
        # Generate embeddings using safe method
        embeddings = self._generate_embeddings_safe(all_chunks)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create optimized index for small dataset
        dimension = embeddings.shape[1]
        
        # For 100-200 pages, use simple flat index (most accurate)
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        self.index.add(embeddings)
        
        # Store data
        self.texts = all_chunks
        self.metadata = all_metadata
        self.doc_count = len(pdf_files)
        self.total_chunks = len(all_chunks)
        
        # Save to disk - NEW FORMAT
        os.makedirs(INDEX_PATH, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(INDEX_PATH, "faiss.index"))
        
        with open(os.path.join(INDEX_PATH, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "doc_count": self.doc_count
            }, f)
        
        logger.info(f"Index built successfully: {self.doc_count} documents, {self.total_chunks} chunks")
        logger.info(f"Average chunks per document: {self.total_chunks / self.doc_count:.1f}")
    
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for query"""
        if self.index is None or self.embedding_model is None:
            logger.warning("Cannot retrieve: index or embedding model not available")
            return []
        
        try:
            logger.debug(f"Retrieving top {k} chunks for query: '{query[:50]}...'")
            
            # Encode query with CPU device
            query_vec = self.embedding_model.encode(
                [query],
                device='cpu',
                convert_to_tensor=False
            ).astype('float32')
            faiss.normalize_L2(query_vec)
            
            # Search
            distances, indices = self.index.search(query_vec, k)
            
            # Build results
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.texts):
                    results.append({
                        "text": self.texts[idx],
                        "metadata": self.metadata[idx],
                        "score": float(distances[0][i]),
                        "rank": i + 1
                    })
            
            # Log retrieval quality
            if results:
                avg_score = sum(r["score"] for r in results) / len(results)
                logger.info(f"Retrieved {len(results)} chunks, avg score: {avg_score:.3f}, top score: {results[0]['score']:.3f}")
            else:
                logger.warning("No chunks retrieved")
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            "documents": self.doc_count,
            "total_chunks": self.total_chunks,
            "avg_chunks_per_doc": self.total_chunks / self.doc_count if self.doc_count > 0 else 0,
            "index_loaded": self.index is not None,
            "embedding_model": self.embedding_model is not None
        }

# ============================================================================
# PUBLIC API
# ============================================================================

_rag_instance: Optional[OptimizedRAGSystem] = None

def get_rag_system() -> OptimizedRAGSystem:
    """Get singleton RAG system"""
    global _rag_instance
    if _rag_instance is None:
        logger.info("Initializing OptimizedRAGSystem...")
        _rag_instance = OptimizedRAGSystem()
    return _rag_instance

def retrieve_and_format_context(query: str, k: int = TOP_K_RETRIEVAL) -> str:
    """Retrieve and format context for query"""
    rag_system = get_rag_system()
    
    # Retrieve chunks
    results = rag_system.retrieve(query, k)
    
    if not results:
        logger.warning("No results retrieved from RAG")
        return ""
    
    # Format chunks for context
    chunks = []
    for res in results:
        source = res['metadata'].get('source', 'Unknown')
        chunk_text = res['text']
        
        # Add source attribution
        formatted_chunk = f"[Source: {source}]\n{chunk_text}"
        chunks.append(formatted_chunk)
    
    # Use formatter to clean and concatenate
    formatted = format_retrieved_context(chunks, max_chars=MAX_CONTEXT_LENGTH)
    
    logger.debug(f"Formatted context: {len(formatted)} chars from {len(results)} chunks")
    
    return formatted

def build_index(pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
    """Build or rebuild the index"""
    rag_system = get_rag_system()
    rag_system.build_index(pdf_directory, force_rebuild)

def get_index_stats() -> Dict[str, Any]:
    """Get RAG system statistics"""
    rag_system = get_rag_system()
    return rag_system.get_index_stats()