#!/usr/bin/env python3
"""
Utility script to build or rebuild the FAISS index for the Pharma Enterprise Assistant.
Run this after adding new PDFs to the data directory.

Usage:
    python build_index.py                  # Build index from data/ folder
    python build_index.py --rebuild        # Force rebuild even if index exists
    python build_index.py --path /custom   # Use custom PDF directory
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
        'faiss',
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
            nltk.download(name)

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

def check_existing_index() -> bool:
    """Check if an index already exists"""
    from config import INDEX_PATH
    
    index_file = Path(INDEX_PATH) / "faiss.index"
    metadata_file = Path(INDEX_PATH) / "metadata.pkl"
    
    if index_file.exists() and metadata_file.exists():
        index_size_mb = index_file.stat().st_size / (1024 * 1024)
        metadata_size_mb = metadata_file.stat().st_size / (1024 * 1024)
        
        logger.info(f"Existing index found:")
        logger.info(f"  - Index: {index_size_mb:.1f} MB")
        logger.info(f"  - Metadata: {metadata_size_mb:.1f} MB")
        return True
    
    return False

def estimate_processing_time(pdf_files: list) -> float:
    """Estimate processing time based on file sizes"""
    total_size_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
    
    # Rough estimates based on typical processing speeds
    # ~10 MB/minute for PDF extraction + embedding generation
    estimated_minutes = total_size_mb / 10
    
    return max(1.0, estimated_minutes)

def build_index(pdf_directory: str, force_rebuild: bool = False):
    """Build the FAISS index"""
    
    # Check if index exists
    if check_existing_index() and not force_rebuild:
        logger.warning("Index already exists!")
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
    logger.info("BUILDING FAISS INDEX")
    logger.info("="*60)
    logger.info("This may take several minutes...")
    logger.info("Processing steps:")
    logger.info("  1. Extracting text from PDFs")
    logger.info("  2. Creating semantic chunks")
    logger.info("  3. Generating embeddings")
    logger.info("  4. Building FAISS index")
    logger.info("  5. Saving to disk")
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
        
        # Display statistics
        stats = get_index_stats()
        logger.info("")
        logger.info("Index Statistics:")
        logger.info(f"  📚 Documents: {stats['documents']}")
        logger.info(f"  📄 Total chunks: {stats['total_chunks']}")
        logger.info(f"  📊 Avg chunks per doc: {stats['avg_chunks_per_doc']:.1f}")
        
        # Verify index files
        from config import INDEX_PATH
        index_path = Path(INDEX_PATH)
        index_file = index_path / "faiss.index"
        metadata_file = index_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            index_size = index_file.stat().st_size / (1024 * 1024)
            metadata_size = metadata_file.stat().st_size / (1024 * 1024)
            
            logger.info("")
            logger.info("Created files:")
            logger.info(f"  📁 {index_file}: {index_size:.1f} MB")
            logger.info(f"  📁 {metadata_file}: {metadata_size:.1f} MB")
        
        logger.info("")
        logger.info("✨ Index is ready to use!")
        logger.info("You can now deploy your application.")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}", exc_info=True)
        logger.error("")
        logger.error("Common issues:")
        logger.error("  - Corrupted PDF files")
        logger.error("  - Insufficient memory")
        logger.error("  - Missing dependencies")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Build FAISS index for Pharma Enterprise Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_index.py                    # Build from data/ folder
  python build_index.py --rebuild          # Force rebuild
  python build_index.py --path docs/       # Use custom directory
  python build_index.py --check-only       # Only check if index exists
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
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PHARMA ENTERPRISE ASSISTANT - INDEX BUILDER")
    logger.info("="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Download NLTK data if needed
    download_nltk_data()
    
    # Check only mode
    if args.check_only:
        if check_existing_index():
            logger.info("✅ Index exists and is ready to use")
            sys.exit(0)
        else:
            logger.warning("❌ No index found - need to build")
            sys.exit(1)
    
    # Build index
    build_index(args.path, args.rebuild)

if __name__ == "__main__":
    main()