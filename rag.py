# rag.py - Pinecone-based RAG for scalable document retrieval

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec

from embeddings import get_embedding_model
from semantic_chunker import SemanticChunker
from context_formatter import format_retrieved_context
from config import (
    PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME,
    PINECONE_DIMENSION, PINECONE_METRIC, PINECONE_BATCH_SIZE,
    PINECONE_NAMESPACE, PDF_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RETRIEVAL, CHUNKING_STRATEGY, MAX_CHUNK_TOKENS,
    EMBEDDING_BATCH_SIZE, MAX_CONTEXT_LENGTH
)

logger = logging.getLogger(__name__)

class PineconeRAGSystem:
    """RAG system using Pinecone for scalable vector search"""
    
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.semantic_chunker = SemanticChunker()
        self.pinecone_index = None
        self.pc = None
        self.doc_count = 0
        self.total_chunks = 0
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY not configured")
            return
        
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric=PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=PINECONE_ENVIRONMENT
                    )
                )
            
            # Connect to index
            self.pinecone_index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Get index stats
            stats = self.pinecone_index.describe_index_stats()
            self.total_chunks = stats.total_vector_count
            
            logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            logger.info(f"Total vectors in index: {self.total_chunks}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}", exc_info=True)
            self.pinecone_index = None
            self.pc = None
    
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
    
    def chunk_document(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create optimized chunks for retrieval with semantic chunking"""
        chunks: List[Dict[str, Any]] = []
        
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
                
                # Generate unique ID for chunk
                chunk_id = self._generate_chunk_id(source, i, chunk_text)
                
                # Create chunk data
                chunk_data = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_index": i,
                        "chunk_size": len(chunk_text),
                        "strategy": chunk_metadata.get("strategy", CHUNKING_STRATEGY),
                        "section": chunk_metadata.get("section", "unknown"),
                        **chunk_metadata
                    }
                }
                
                chunks.append(chunk_data)
                
            logger.info(f"Created {len(chunks)} chunks from {source} using {CHUNKING_STRATEGY} strategy")
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}. Using fallback.")
            # Fallback to simple overlapping chunks
            chunks = self._fallback_chunking(text, source)
        
        return chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean chunk text for better retrieval"""
        import re
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove page markers if at the beginning
        text = re.sub(r'^\[Page \d+\]\s*', '', text)
        
        return text.strip()
    
    def _generate_chunk_id(self, source: str, index: int, text: str) -> str:
        """Generate unique ID for chunk"""
        # Create deterministic ID based on source, index, and content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{source}_{index}_{content_hash}"
    
    def _fallback_chunking(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Simple overlapping chunk strategy as fallback"""
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + CHUNK_SIZE].strip()
            if len(chunk_text) > 100:
                chunk_id = self._generate_chunk_id(source, i // step, chunk_text)
                chunk_data = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_index": i // step,
                        "strategy": "fallback",
                        "chunk_size": len(chunk_text)
                    }
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches for efficiency"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")
        
        # Disable multiprocessing for stability
        torch.set_num_threads(1)
        
        embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i+EMBEDDING_BATCH_SIZE]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_tensor=False,
                device='cpu',
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            embeddings.append(batch_embeddings)
            
            # Log progress
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(texts)} embeddings")
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings).astype('float32')
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def build_index(self, pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
        """Build Pinecone index from PDF documents"""
        if not self.pinecone_index:
            logger.error("Pinecone not initialized. Cannot build index.")
            return
        
        if self.embedding_model is None:
            logger.error("Cannot build index: embedding model not loaded.")
            return
        
        # Check if index already has data
        stats = self.pinecone_index.describe_index_stats()
        if stats.total_vector_count > 0 and not force_rebuild:
            logger.info(f"Index already contains {stats.total_vector_count} vectors. Use force_rebuild=True to rebuild.")
            return
        
        # Clear index if force rebuild
        if force_rebuild and stats.total_vector_count > 0:
            logger.info("Clearing existing vectors from index...")
            self.pinecone_index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        
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
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            logger.info(f"Processing: {pdf_file}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_file}")
                continue
            
            # Create chunks with semantic chunking
            chunks = self.chunk_document(text, pdf_file)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.error("No chunks created from PDFs")
            return
        
        logger.info(f"Created {len(all_chunks)} total chunks from {len(pdf_files)} documents")
        
        # Prepare vectors for Pinecone
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self._generate_embeddings_batch(chunk_texts)
        
        # Prepare data for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            vector = {
                "id": chunk["id"],
                "values": embedding.tolist(),
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"][:1000]  # Store first 1000 chars in metadata for retrieval
                }
            }
            vectors.append(vector)
        
        # Upsert to Pinecone in batches
        logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
        for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
            batch = vectors[i:i+PINECONE_BATCH_SIZE]
            self.pinecone_index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
            logger.info(f"Upserted batch {i//PINECONE_BATCH_SIZE + 1}/{(len(vectors)-1)//PINECONE_BATCH_SIZE + 1}")
        
        # Update stats
        self.doc_count = len(pdf_files)
        self.total_chunks = len(all_chunks)
        
        logger.info(f"Index built successfully: {self.doc_count} documents, {self.total_chunks} chunks")
        logger.info(f"Average chunks per document: {self.total_chunks / self.doc_count:.1f}")
    
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for query from Pinecone"""
        if not self.pinecone_index or not self.embedding_model:
            logger.warning("Cannot retrieve: Pinecone or embedding model not available")
            return []
        
        try:
            logger.debug(f"Retrieving top {k} chunks for query: '{query[:50]}...'")
            
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                device='cpu',
                convert_to_tensor=False,
                normalize_embeddings=True
            ).astype('float32')[0]
            
            # Search in Pinecone
            response = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=True,
                namespace=PINECONE_NAMESPACE
            )
            
            # Build results
            results = []
            for match in response.matches:
                # Retrieve full text from metadata or reconstruct if needed
                text = match.metadata.get("text", "")
                
                results.append({
                    "text": text,
                    "metadata": {
                        "source": match.metadata.get("source", "unknown"),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                        "strategy": match.metadata.get("strategy", "unknown"),
                        "section": match.metadata.get("section", "unknown")
                    },
                    "score": float(match.score),
                    "rank": len(results) + 1
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
        """Get statistics about the Pinecone index"""
        if self.pinecone_index:
            try:
                stats = self.pinecone_index.describe_index_stats()
                return {
                    "documents": self.doc_count,
                    "total_chunks": stats.total_vector_count,
                    "avg_chunks_per_doc": stats.total_vector_count / self.doc_count if self.doc_count > 0 else 0,
                    "index_loaded": True,
                    "embedding_model": self.embedding_model is not None,
                    "namespaces": list(stats.namespaces.keys()),
                    "dimension": stats.dimension
                }
            except Exception as e:
                logger.error(f"Failed to get index stats: {e}")
        
        return {
            "documents": 0,
            "total_chunks": 0,
            "avg_chunks_per_doc": 0,
            "index_loaded": False,
            "embedding_model": self.embedding_model is not None
        }

# ============================================================================
# PUBLIC API
# ============================================================================

_rag_instance: Optional[PineconeRAGSystem] = None

def get_rag_system() -> PineconeRAGSystem:
    """Get singleton RAG system"""
    global _rag_instance
    if _rag_instance is None:
        logger.info("Initializing PineconeRAGSystem...")
        _rag_instance = PineconeRAGSystem()
    return _rag_instance

def enhance_query_for_retrieval(query: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """
    Simple query enhancement for better retrieval of follow-up questions
    """
    query_lower = query.lower()
    
    # Check if this looks like a follow-up
    follow_up_indicators = [
        "which ones", "what about", "those", "they", "them", 
        "most common", "most severe", "worst", "best", "serious"
    ]
    
    is_followup = any(indicator in query_lower for indicator in follow_up_indicators)
    
    if not is_followup or not conversation_history:
        return query
    
    # Look for recent medical topics in the last few messages
    medical_keywords = []
    for msg in conversation_history[-4:]:  # Last 2 exchanges
        content = msg.get("content", "").lower()
        if "side effect" in content:
            medical_keywords.append("side effects")
        if "dosage" in content or "dose" in content:
            medical_keywords.append("dosage")
        if "interaction" in content:
            medical_keywords.append("interactions")
        if "journvax" in content:
            medical_keywords.append("Journvax")
    
    # If we found relevant context, enhance the query
    if medical_keywords:
        # Add the most relevant keyword to the query
        if "side effect" in " ".join(medical_keywords):
            enhanced = f"{query} Journvax side effects"
        elif "dosage" in " ".join(medical_keywords):
            enhanced = f"{query} Journvax dosage"
        else:
            enhanced = f"{query} Journvax"
        
        logger.info(f"Enhanced query for retrieval: '{query}' -> '{enhanced}'")
        return enhanced
    
    return query

def retrieve_and_format_context(query: str, k: int = TOP_K_RETRIEVAL, 
                                conversation_history: List[Dict[str, str]] = None) -> str:
    """Retrieve and format context for query with follow-up handling"""
    rag_system = get_rag_system()
    
    # Enhance query if it looks like a follow-up
    enhanced_query = enhance_query_for_retrieval(query, conversation_history)
    
    # Retrieve chunks using enhanced query
    results = rag_system.retrieve(enhanced_query, k)
    
    if not results:
        logger.warning("No results retrieved from RAG")
        return ""
    
    # Format chunks for context
    chunks = []
    for res in results:
        source = res['metadata'].get('source', 'Unknown')
        chunk_text = res['text']
        
        # Add source attribution and section if available
        section = res['metadata'].get('section', '')
        if section and section != 'unknown':
            formatted_chunk = f"[Source: {source} - {section}]\n{chunk_text}"
        else:
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