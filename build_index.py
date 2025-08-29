#!/usr/bin/env python3
"""
Utility script to build or rebuild the Pinecone index for the Pharma Enterprise Assistant.
Run this after adding new PDFs to the data directory.

Usage:
    python build_index.py                  # Build index from data/ folder
    python build_index.py --rebuild        # Force rebuild even if index exists
    python build_index.py --path /custom   # Use custom PDF directory
    python build_index.py --migrate       # Migrate from FAISS to Pinecone
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check that all required dependencies are installed"""
    required_modules = [
        'pinecone',
        'sentence_transformers',
        'fitz',
        'nltk',
        'numpy'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("All dependencies verified")

def check_api_keys():
    """Check that required API keys are configured"""
    from config import PINECONE_API_KEY, ANTHROPIC_API_KEY
    
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not configured!")
        logger.error("Please set PINECONE_API_KEY environment variable or add to .env file")
        sys.exit(1)
    
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not configured - needed for runtime but not index building")
    
    logger.info("API keys verified")

def download_nltk_data():
    """Download required NLTK data if not present"""
    import nltk
    
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK data: {name}")
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {name}: {e}")

def validate_pdf_directory(pdf_dir: str) -> list:
    """Validate PDF directory and return list of PDF files"""
    data_path = Path(pdf_dir)
    
    if not data_path.exists():
        logger.error(f"Directory '{pdf_dir}' not found!")
        logger.info(f"Please create the directory and add PDF files")
        sys.exit(1)
    
    if not data_path.is_dir():
        logger.error(f"'{pdf_dir}' is not a directory!")
        sys.exit(1)
    
    # Find PDF files
    pdf_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in '{pdf_dir}'")
        logger.info("Please add PDF files to the directory")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF files:")
    for pdf in sorted(pdf_files):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        logger.info(f"  - {pdf.name} ({size_mb:.1f} MB)")
    
    return pdf_files

def check_existing_pinecone_index() -> int:
    """Check if Pinecone index already exists and return vector count"""
    try:
        from rag import get_rag_system
        rag_system = get_rag_system()
        
        if rag_system.pinecone_index:
            stats = rag_system.get_index_stats()
            vector_count = stats.get("total_chunks", 0)
            
            if vector_count > 0:
                logger.info(f"Existing Pinecone index found with {vector_count} vectors")
                return vector_count
    except Exception as e:
        logger.debug(f"Could not check existing index: {e}")
    
    return 0

def migrate_from_faiss():
    """Migrate existing FAISS index to Pinecone"""
    from config import INDEX_PATH
    import pickle
    
    faiss_index_file = Path(INDEX_PATH) / "faiss.index"
    metadata_file = Path(INDEX_PATH) / "metadata.pkl"
    
    if not (faiss_index_file.exists() and metadata_file.exists()):
        logger.error("No FAISS index found to migrate")
        return False
    
    logger.info("Found FAISS index to migrate")
    
    try:
        # Load FAISS metadata
        with open(metadata_file, "rb") as f:
            saved_data = pickle.load(f)
        
        if isinstance(saved_data, tuple):
            texts, metadata = saved_data
            logger.info(f"Loaded {len(texts)} chunks from FAISS index")
        elif isinstance(saved_data, dict):
            texts = saved_data["texts"]
            metadata = saved_data["metadata"]
            logger.info(f"Loaded {len(texts)} chunks from FAISS index")
        else:
            logger.error("Unknown FAISS metadata format")
            return False
        
        # Initialize Pinecone RAG system
        from rag import get_rag_system
        rag_system = get_rag_system()
        
        if not rag_system.pinecone_index:
            logger.error("Failed to initialize Pinecone")
            return False
        
        # Generate embeddings and upload to Pinecone
        logger.info("Migrating chunks to Pinecone...")
        
        from config import PINECONE_BATCH_SIZE, PINECONE_NAMESPACE
        import hashlib
        
        # Generate embeddings
        embeddings = rag_system._generate_embeddings_batch(texts)
        
        # Prepare vectors
        vectors = []
        for i, (text, meta, embedding) in enumerate(zip(texts, metadata, embeddings)):
            # Generate ID
            chunk_id = f"migrated_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
            
            vector = {
                "id": chunk_id,
                "values": embedding.tolist(),
                "metadata": {
                    **meta,
                    "text": text[:1000],  # Store first 1000 chars
                    "migrated_from_faiss": True
                }
            }
            vectors.append(vector)
        
        # Upsert in batches
        for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
            batch = vectors[i:i+PINECONE_BATCH_SIZE]
            rag_system.pinecone_index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
            logger.info(f"Migrated batch {i//PINECONE_BATCH_SIZE + 1}")
        
        logger.info(f"Successfully migrated {len(vectors)} vectors to Pinecone")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False

def estimate_processing_time(pdf_files: list) -> float:
    """Estimate processing time based on file sizes"""
    total_size_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
    
    # Rough estimates based on typical processing speeds
    # ~5 MB/minute for PDF extraction + embedding generation + Pinecone upload
    estimated_minutes = total_size_mb / 5
    
    return max(1.0, estimated_minutes)

def build_index(pdf_directory: str, force_rebuild: bool = False):
    """Build the Pinecone index"""
    
    # Check if index exists
    vector_count = check_existing_pinecone_index()
    if vector_count > 0 and not force_rebuild:
        logger.warning(f"Index already exists with {vector_count} vectors!")
        logger.info("Use --rebuild flag to force rebuild")
        response = input("Do you want to rebuild anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return
    
    # Validate PDFs
    pdf_files = validate_pdf_directory(pdf_directory)
    
    # Estimate time
    estimated_time = estimate_processing_time(pdf_files)
    logger.info(f"Estimated processing time: {estimated_time:.1f} minutes")
    
    # Import RAG system
    try:
        logger.info("Loading RAG system...")
        from rag import build_index as rag_build_index, get_index_stats
    except ImportError as e:
        logger.error(f"Failed to import RAG system: {e}")
        logger.info("Make sure all files are in place")
        sys.exit(1)
    
    # Build index
    logger.info("="*60)
    logger.info("BUILDING PINECONE INDEX")
    logger.info("="*60)
    logger.info("This may take several minutes...")
    logger.info("Processing steps:")
    logger.info("  1. Extracting text from PDFs")
    logger.info("  2. Creating semantic chunks (hybrid strategy)")
    logger.info("  3. Generating embeddings")
    logger.info("  4. Uploading to Pinecone cloud")
    logger.info("  5. Verifying index")
    logger.info("")
    
    start_time = time.time()
    
    try:
        rag_build_index(pdf_directory, force_rebuild=True)
        
        elapsed_time = time.time() - start_time
        logger.info("")
        logger.info("="*60)
        logger.info(f"✅ INDEX BUILD SUCCESSFUL")
        logger.info(f"Time taken: {elapsed_time/60:.1f} minutes")
        logger.info("="*60)
        
        # Display statistics with better error handling
        stats = get_index_stats()
        logger.info("")
        logger.info("Index Statistics:")
        logger.info(f"  📚 Documents: {stats.get('documents', 0)}")
        logger.info(f"  📄 Total chunks: {stats.get('total_chunks', 0)}")
        
        # Safely handle average calculation
        if stats.get('documents', 0) > 0:
            avg_chunks = stats.get('total_chunks', 0) / stats.get('documents', 1)
            logger.info(f"  📊 Avg chunks per doc: {avg_chunks:.1f}")
        else:
            logger.info(f"  📊 Avg chunks per doc: 0")
        
        # Safely handle namespaces display
        namespaces = stats.get('namespaces', [])
        if namespaces:
            if isinstance(namespaces, list) and len(namespaces) > 0:
                logger.info(f"  🌐 Namespace: {namespaces[0]}")
            elif isinstance(namespaces, dict):
                # Sometimes Pinecone returns namespaces as a dict
                namespace_names = list(namespaces.keys())
                if namespace_names:
                    logger.info(f"  🌐 Namespace: {namespace_names[0]}")
            else:
                logger.info(f"  🌐 Namespace: default")
        else:
            logger.info(f"  🌐 Namespace: default")
        
        # Display dimension if available
        if 'dimension' in stats:
            logger.info(f"  🔢 Embedding Dimension: {stats['dimension']}")
        
        logger.info("")
        logger.info("✨ Pinecone index is ready to use!")
        logger.info("You can now run the application with: streamlit run app.py")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}", exc_info=True)
        logger.error("")
        logger.error("Common issues:")
        logger.error("  - Invalid Pinecone API key")
        logger.error("  - Network connectivity issues")
        logger.error("  - Corrupted PDF files")
        logger.error("  - Insufficient memory")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Build Pinecone index for Pharma Enterprise Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_index.py                    # Build from data/ folder
  python build_index.py --rebuild          # Force rebuild
  python build_index.py --path docs/       # Use custom directory
  python build_index.py --check-only       # Only check if index exists
  python build_index.py --migrate          # Migrate from FAISS to Pinecone
        """
    )
    
    parser.add_argument(
        '--path', 
        default='data',
        help='Path to PDF directory (default: data)'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if index exists, don\'t build'
    )
    
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Migrate existing FAISS index to Pinecone'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PHARMA ENTERPRISE ASSISTANT - PINECONE INDEX BUILDER")
    logger.info("="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Check API keys
    check_api_keys()
    
    # Download NLTK data if needed
    download_nltk_data()
    
    # Migrate mode
    if args.migrate:
        logger.info("Starting FAISS to Pinecone migration...")
        success = migrate_from_faiss()
        if success:
            logger.info("✅ Migration completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Migration failed")
            sys.exit(1)
    
    # Check only mode
    if args.check_only:
        vector_count = check_existing_pinecone_index()
        if vector_count > 0:
            logger.info(f"✅ Pinecone index exists with {vector_count} vectors")
            sys.exit(0)
        else:
            logger.warning("❌ No vectors found in Pinecone index")
            sys.exit(1)
    
    # Build index
    build_index(args.path, args.rebuild)

if __name__ == "__main__":
    main()