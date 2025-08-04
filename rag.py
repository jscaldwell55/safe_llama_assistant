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
    MAX_CHUNK_TOKENS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Simplified RAG system that trusts the model while maintaining good retrieval.
    """
    
    def __init__(self):
        # Initialize embedding model with proper environment handling
        self._init_embedding_model()
        self.semantic_chunker = SemanticChunker()
        self.index = None
        self.texts = []
        self.metadata = []
        self._load_index()
    
    def _init_embedding_model(self):
        """Initialize embedding model, handling HF endpoint conflicts."""
        try:
            import os
            # Temporarily clear HF_ENDPOINT to avoid conflicts
            original_endpoint = os.environ.pop('HF_ENDPOINT', None)
            
            try:
                # This will download from public HuggingFace hub
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL_NAME}")
            finally:
                # Restore original endpoint if it existed
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                    
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("RAG system will work with limited functionality")
            logger.info("You may need to download the model manually or check your internet connection")
            self.embedding_model = None
    
    def _load_index(self):
        """Load the FAISS index and metadata if they exist."""
        try:
            index_file = os.path.join(INDEX_PATH, "faiss.index")
            metadata_file = os.path.join(INDEX_PATH, "metadata.pkl")
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                self.index = faiss.read_index(index_file)
                with open(metadata_file, "rb") as f:
                    self.texts, self.metadata = pickle.load(f)
                logger.info(f"Loaded index with {len(self.texts)} chunks")
            else:
                logger.warning("No existing index found. Run build_index() to create one.")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.index = None
            self.texts = []
            self.metadata = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text.append(text)
            
            doc.close()
            return "\n\n".join(full_text)
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def chunk_document(self, text: str, source: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Chunk document using semantic chunking.
        Simpler approach that preserves meaning.
        """
        chunks = []
        
        try:
            # Use semantic chunker
            semantic_chunks = self.semantic_chunker.semantic_chunk(
                text, 
                strategy=CHUNKING_STRATEGY,
                max_tokens=MAX_CHUNK_TOKENS
            )
            
            for i, (chunk_text, chunk_metadata) in enumerate(semantic_chunks):
                # Skip very short chunks
                if len(chunk_text.strip()) < 50:
                    continue
                
                # Simple metadata
                metadata = {
                    "source": source,
                    "chunk_id": i,
                    **chunk_metadata
                }
                
                chunks.append((chunk_text.strip(), metadata))
                
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}. Using fallback.")
            # Simple fallback chunking
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = text[i:i + CHUNK_SIZE].strip()
                if len(chunk_text) > 50:
                    metadata = {
                        "source": source,
                        "chunk_id": i // (CHUNK_SIZE - CHUNK_OVERLAP),
                        "strategy": "fallback"
                    }
                    chunks.append((chunk_text, metadata))
        
        return chunks
    
    def build_index(self, pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
        """
        Build FAISS index from PDF files.
        """
        if self.index is not None and not force_rebuild:
            logger.info("Index already loaded. Use force_rebuild=True to rebuild.")
            return
        
        if not os.path.exists(pdf_directory):
            logger.error(f"PDF directory {pdf_directory} does not exist")
            return
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            # Try to create sample data
            self._create_sample_data(pdf_directory)
            return
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        all_chunks = []
        all_metadata = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            logger.info(f"Processing {pdf_file}...")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                continue
            
            # Chunk the document
            chunks = self.chunk_document(text, pdf_file)
            
            for chunk_text, metadata in chunks:
                all_chunks.append(chunk_text)
                all_metadata.append(metadata)
        
        if not all_chunks:
            logger.error("No text chunks extracted from PDFs")
            return
        
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        os.makedirs(INDEX_PATH, exist_ok=True)
        faiss.write_index(index, os.path.join(INDEX_PATH, "faiss.index"))
        
        with open(os.path.join(INDEX_PATH, "metadata.pkl"), "wb") as f:
            pickle.dump((all_chunks, all_metadata), f)
        
        self.index = index
        self.texts = all_chunks
        self.metadata = all_metadata
        
        logger.info(f"Index built successfully with {len(all_chunks)} chunks")
    
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Simple retrieval - let the model figure out what's relevant.
        """
        if self.index is None or len(self.texts) == 0:
            logger.warning("No index available for retrieval")
            return []
        
        try:
            # Encode query
            query_vector = self.embedding_model.encode([query])
            
            # Search
            distances, indices = self.index.search(query_vector.astype('float32'), k)
            
            # Return results
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.texts):
                    results.append({
                        "text": self.texts[idx],
                        "metadata": self.metadata[idx],
                        "score": float(distances[0][i])
                    })
            
            logger.info(f"Retrieved {len(results)} chunks for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def _create_sample_data(self, directory: str):
        """Create sample data if no PDFs exist."""
        os.makedirs(directory, exist_ok=True)
        
        sample_content = """
        # Sample Medical Document
        
        ## Drug Information
        
        This is a sample document for testing the RAG system.
        
        ### Dosage and Administration
        The recommended dosage varies by condition and patient factors.
        Always consult healthcare professionals for medical advice.
        
        ### Side Effects
        Common side effects may include mild symptoms.
        Serious side effects are rare but require immediate medical attention.
        
        ### Warnings
        This information is for educational purposes only.
        Not intended as medical advice.
        """
        
        # Create a simple text file as fallback
        with open(os.path.join(directory, "sample_data.txt"), "w") as f:
            f.write(sample_content)
        
        logger.info("Created sample data file")

# Global RAG system instance
rag_system = RAGSystem()

# Convenience functions for backward compatibility
def retrieve(query: str, k: int = TOP_K_RETRIEVAL) -> List[str]:
    """Simple retrieval function."""
    results = rag_system.retrieve(query, k)
    return [result["text"] for result in results]

def build_index(pdf_directory: str = PDF_DATA_PATH, force_rebuild: bool = False):
    """Build the index."""
    return rag_system.build_index(pdf_directory, force_rebuild)