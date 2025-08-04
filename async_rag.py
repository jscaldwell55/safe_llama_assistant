import faiss
import pickle
import os
import logging
import asyncio
import aiofiles
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
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncRAGSystem:
    """
    Async RAG system with optimized batch processing and async I/O operations.
    """
    
    def __init__(self):
        # Initialize embedding model with proper environment handling
        self._init_embedding_model()
        self.semantic_chunker = SemanticChunker()
        self.index = None
        self.texts = []
        self.metadata = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
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
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    async def _load_index(self):
        """Load existing FAISS index and metadata asynchronously if available."""
        index_file = os.path.join(INDEX_PATH, "faiss.index")
        metadata_file = os.path.join(INDEX_PATH, "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                # FAISS operations are CPU-bound, so use thread executor
                loop = asyncio.get_event_loop()
                self.index = await loop.run_in_executor(
                    self.executor, 
                    faiss.read_index, 
                    index_file
                )
                
                async with aiofiles.open(metadata_file, "rb") as f:
                    content = await f.read()
                    self.texts, self.metadata = pickle.loads(content)
                
                logger.info(f"Loaded existing index with {len(self.texts)} chunks")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self.index = None
                self.texts = []
                self.metadata = []
    
    async def initialize(self):
        """Initialize the RAG system asynchronously."""
        await self._load_index()
    
    async def _extract_text_from_pdf(self, pdf_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Extract text from a single PDF file asynchronously."""
        loop = asyncio.get_event_loop()
        
        def extract_sync():
            chunks = []
            metadata = []
            
            try:
                doc = fitz.open(pdf_path)
                file_name = os.path.basename(pdf_path)
                
                if CHUNKING_STRATEGY == "hybrid":
                    # Let semantic chunker handle the entire document
                    full_text = ""
                    for page_num, page in enumerate(doc):
                        page_text = page.get_text()
                        if page_text.strip():
                            full_text += f"\n[Page {page_num + 1}]\n{page_text}"
                    
                    if full_text.strip():
                        semantic_chunks = self.semantic_chunker.chunk_text(full_text)
                        for chunk in semantic_chunks:
                            chunks.append(chunk)
                            metadata.append({
                                "source": file_name,
                                "type": "semantic_chunk"
                            })
                else:
                    # Original page-based chunking
                    for page_num, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            if len(text) > CHUNK_SIZE:
                                # Split large pages into smaller chunks
                                words = text.split()
                                for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                                    chunk = " ".join(words[i:i + CHUNK_SIZE])
                                    chunks.append(chunk)
                                    metadata.append({
                                        "source": file_name,
                                        "page": page_num + 1,
                                        "chunk_index": i // (CHUNK_SIZE - CHUNK_OVERLAP)
                                    })
                            else:
                                chunks.append(text)
                                metadata.append({
                                    "source": file_name,
                                    "page": page_num + 1
                                })
                
                doc.close()
                logger.info(f"Extracted {len(chunks)} chunks from {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
            
            return chunks, metadata
        
        return await loop.run_in_executor(self.executor, extract_sync)
    
    async def _process_embeddings_batch(self, chunks: List[str]) -> np.ndarray:
        """Process embeddings for a batch of chunks asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.embedding_model.encode,
            chunks,
            False  # show_progress_bar=False
        )
    
    async def build_index(self):
        """Build FAISS index from PDF files in the data directory using async operations."""
        if not os.path.exists(PDF_DATA_PATH):
            logger.error(f"Data path {PDF_DATA_PATH} does not exist")
            return
        
        pdf_files = [f for f in os.listdir(PDF_DATA_PATH) if f.endswith('.pdf')]
        if not pdf_files:
            logger.error(f"No PDF files found in {PDF_DATA_PATH}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Extract text from all PDFs concurrently
        extraction_tasks = [
            self._extract_text_from_pdf(os.path.join(PDF_DATA_PATH, pdf_file))
            for pdf_file in pdf_files
        ]
        
        results = await asyncio.gather(*extraction_tasks)
        
        # Combine results
        all_chunks = []
        all_metadata = []
        for chunks, metadata in results:
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)
        
        if not all_chunks:
            logger.error("No text chunks extracted from PDFs")
            return
        
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks using batch size {EMBEDDING_BATCH_SIZE}...")
        
        # Process embeddings in batches asynchronously
        embedding_tasks = []
        for i in range(0, len(all_chunks), EMBEDDING_BATCH_SIZE):
            batch_end = min(i + EMBEDDING_BATCH_SIZE, len(all_chunks))
            batch_chunks = all_chunks[i:batch_end]
            logger.info(f"Preparing batch {i//EMBEDDING_BATCH_SIZE + 1}/{(len(all_chunks) + EMBEDDING_BATCH_SIZE - 1)//EMBEDDING_BATCH_SIZE}")
            embedding_tasks.append(self._process_embeddings_batch(batch_chunks))
        
        # Process all batches concurrently (with some limit to avoid overwhelming the system)
        max_concurrent = 3
        all_embeddings = []
        
        for i in range(0, len(embedding_tasks), max_concurrent):
            batch_tasks = embedding_tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch_tasks)
            all_embeddings.extend(batch_results)
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        loop = asyncio.get_event_loop()
        dimension = embeddings.shape[1]
        index = await loop.run_in_executor(
            self.executor,
            faiss.IndexFlatL2,
            dimension
        )
        await loop.run_in_executor(
            self.executor,
            index.add,
            embeddings.astype('float32')
        )
        
        # Save index and metadata asynchronously
        os.makedirs(INDEX_PATH, exist_ok=True)
        
        index_file = os.path.join(INDEX_PATH, "faiss.index")
        await loop.run_in_executor(
            self.executor,
            faiss.write_index,
            index,
            index_file
        )
        
        metadata_file = os.path.join(INDEX_PATH, "metadata.pkl")
        async with aiofiles.open(metadata_file, "wb") as f:
            await f.write(pickle.dumps((all_chunks, all_metadata)))
        
        self.index = index
        self.texts = all_chunks
        self.metadata = all_metadata
        
        logger.info(f"Index built successfully with {len(all_chunks)} chunks")
    
    async def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents asynchronously.
        """
        if self.index is None or len(self.texts) == 0:
            logger.warning("No index available for retrieval")
            return []
        
        try:
            # Encode query asynchronously
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                self.executor,
                self.embedding_model.encode,
                [query]
            )
            
            # Search in FAISS index
            distances, indices = await loop.run_in_executor(
                self.executor,
                self.index.search,
                query_embedding.astype('float32'),
                min(k, len(self.texts))
            )
            
            # Prepare results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.texts):
                    results.append({
                        "text": self.texts[idx],
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                        "score": float(distance)
                    })
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)