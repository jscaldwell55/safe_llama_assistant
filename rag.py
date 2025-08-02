import faiss
import pickle
import os
import logging
import fitz  # PyMuPDF
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL_NAME, 
    INDEX_PATH, 
    PDF_DATA_PATH, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    TOP_K_RETRIEVAL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system for document search and retrieval.
    """
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None
        self.texts = []
        self.metadata = []
        self._load_index()
    
    def _load_index(self):
        """Load the FAISS index and metadata if they exist."""
        try:
            if os.path.exists(f"{INDEX_PATH}/faiss.index") and os.path.exists(f"{INDEX_PATH}/metadata.pkl"):
                self.index = faiss.read_index(f"{INDEX_PATH}/faiss.index")
                with open(f"{INDEX_PATH}/metadata.pkl", "rb") as f:
                    self.texts, self.metadata = pickle.load(f)
                logger.info(f"Loaded index with {len(self.texts)} chunks")
            else:
                logger.warning("No existing index found. Run build_index() to create one.")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.index = None
            self.texts = []
            self.metadata = []
    
    def extract_chunks_from_pdf(self, pdf_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract text chunks from a PDF file with improved error handling.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of (text_chunk, metadata) tuples
        """
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    
                    if not text.strip():
                        continue
                    
                    # Create overlapping chunks
                    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                        chunk_text = text[i:i + CHUNK_SIZE]
                        
                        if len(chunk_text.strip()) < 50:  # Skip very short chunks
                            continue
                        
                        metadata = {
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "chunk_id": len(chunks),
                            "char_start": i,
                            "char_end": min(i + CHUNK_SIZE, len(text))
                        }
                        
                        chunks.append((chunk_text.strip(), metadata))
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num} in {pdf_path}: {e}")
                    continue
            
            doc.close()
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
        
        return chunks
    
    def build_index(self, pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
        """
        Build FAISS index from PDF files in the specified directory.
        
        Args:
            pdf_directory (str): Directory containing PDF files
            force_rebuild (bool): Whether to rebuild even if index exists
        """
        if self.index is not None and not force_rebuild:
            logger.info("Index already loaded. Use force_rebuild=True to rebuild.")
            return
        
        texts = []
        metadata = []
        
        if not os.path.exists(pdf_directory):
            logger.error(f"PDF directory {pdf_directory} does not exist")
            return
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            chunks = self.extract_chunks_from_pdf(pdf_path)
            
            for text_chunk, meta in chunks:
                # Basic deduplication - skip if very similar text exists
                if not self._is_duplicate(text_chunk, texts):
                    texts.append(text_chunk)
                    metadata.append(meta)
        
        if not texts:
            logger.error("No text chunks extracted from PDFs")
            return
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        os.makedirs(INDEX_PATH, exist_ok=True)
        faiss.write_index(index, f"{INDEX_PATH}/faiss.index")
        
        with open(f"{INDEX_PATH}/metadata.pkl", "wb") as f:
            pickle.dump((texts, metadata), f)
        
        self.index = index
        self.texts = texts
        self.metadata = metadata
        
        logger.info(f"Index built successfully with {len(texts)} chunks")
    
    def _is_duplicate(self, new_text: str, existing_texts: List[str], threshold: float = 0.9) -> bool:
        """Check if text is substantially similar to existing texts."""
        new_text_lower = new_text.lower().strip()
        
        for existing_text in existing_texts[-100:]:  # Check only recent texts for efficiency
            existing_text_lower = existing_text.lower().strip()
            
            # Simple similarity check
            if len(new_text_lower) > 0 and len(existing_text_lower) > 0:
                common_chars = len(set(new_text_lower) & set(existing_text_lower))
                similarity = common_chars / max(len(set(new_text_lower)), len(set(existing_text_lower)))
                
                if similarity > threshold:
                    return True
        
        return False
    
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[str]:
        """
        Retrieve top-k most relevant text chunks for a query.
        
        Args:
            query (str): The search query
            k (int): Number of chunks to retrieve
            
        Returns:
            List[str]: List of relevant text chunks
        """
        if self.index is None:
            logger.error("No index loaded. Please build the index first.")
            return []
        
        try:
            query_vector = self.embedding_model.encode([query])
            distances, indices = self.index.search(query_vector.astype('float32'), k)
            
            retrieved_chunks = []
            for idx in indices[0]:
                if 0 <= idx < len(self.texts):
                    retrieved_chunks.append(self.texts[idx])
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query[:50]}...")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def get_chunk_metadata(self, chunk_text: str) -> Dict[str, Any]:
        """Get metadata for a specific chunk."""
        try:
            for i, text in enumerate(self.texts):
                if text == chunk_text:
                    return self.metadata[i]
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
        
        return {}

# Global RAG system instance
rag_system = RAGSystem()

def retrieve(query: str, k: int = TOP_K_RETRIEVAL) -> List[str]:
    """
    Convenience function for retrieving relevant chunks.
    
    Args:
        query (str): The search query
        k (int): Number of chunks to retrieve
        
    Returns:
        List[str]: List of relevant text chunks
    """
    return rag_system.retrieve(query, k)

def build_index(pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
    """
    Convenience function for building the index.
    
    Args:
        pdf_directory (str): Directory containing PDF files
        force_rebuild (bool): Whether to rebuild even if index exists
    """
    return rag_system.build_index(pdf_directory, force_rebuild)
