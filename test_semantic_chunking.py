#!/usr/bin/env python3
"""
Test script for semantic chunking functionality.
"""

import os
import logging
from semantic_chunker import SemanticChunker

# Set environment variable to avoid Streamlit secrets issue
os.environ["HF_TOKEN"] = "dummy_token"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_semantic_chunker():
    """Test the semantic chunker with sample medical text."""
    
    # Sample FDA drug label text
    sample_text = """
HIGHLIGHTS OF PRESCRIBING INFORMATION

1. INDICATIONS AND USAGE

This medication is indicated for the treatment of hypertension in adults to lower blood pressure. Lowering blood pressure reduces the risk of fatal and nonfatal cardiovascular events, primarily strokes and myocardial infarctions.

2. DOSAGE AND ADMINISTRATION

The recommended starting dose is 5 mg once daily. The dose may be increased to 10 mg once daily based on individual patient response. 

Administer with or without food. Swallow tablets whole with water.

3. CONTRAINDICATIONS

This medication is contraindicated in patients with:
• Known hypersensitivity to the active ingredient
• Severe hepatic impairment
• Pregnancy

4. WARNINGS AND PRECAUTIONS

4.1 Hypotension
Monitor blood pressure regularly during treatment initiation and dose adjustments.

4.2 Hepatic Impairment
Use with caution in patients with mild to moderate hepatic impairment.

5. ADVERSE REACTIONS

The most common adverse reactions (≥2%) reported in clinical trials were:
• Headache (8.2%)
• Dizziness (5.4%) 
• Fatigue (3.1%)
• Nausea (2.8%)

6. DRUG INTERACTIONS

6.1 CYP3A4 Inhibitors
Concomitant use with strong CYP3A4 inhibitors may increase drug exposure.

6.2 Antacids
Antacids may reduce absorption when taken within 2 hours.
"""

    logger.info("Testing semantic chunker...")
    chunker = SemanticChunker()
    
    # Test different strategies
    strategies = ["sections", "paragraphs", "sentences", "hybrid", "recursive"]
    
    for strategy in strategies:
        logger.info(f"\n--- Testing strategy: {strategy} ---")
        try:
            chunks = chunker.semantic_chunk(sample_text, strategy=strategy, max_tokens=300)
            logger.info(f"Generated {len(chunks)} chunks")
            
            for i, (chunk_text, metadata) in enumerate(chunks[:3]):  # Show first 3 chunks
                logger.info(f"Chunk {i+1} (len={len(chunk_text)}):")
                logger.info(f"Metadata: {metadata}")
                logger.info(f"Text preview: {chunk_text[:100]}...")
                logger.info("---")
                
        except Exception as e:
            logger.error(f"Error with strategy {strategy}: {e}")

def test_content_classification():
    """Test content type classification."""
    logger.info("\n=== Testing Content Classification ===")
    
    chunker = SemanticChunker()
    
    test_texts = [
        "This medication is indicated for the treatment of hypertension",
        "Follow standard operating procedure for sample collection",
        "Clinical trials demonstrated significant efficacy improvement",
        "FDA approval was granted following regulatory review",
        "General information about the product"
    ]
    
    for text in test_texts:
        content_type = chunker.classify_content_type(text)
        logger.info(f"'{text[:50]}...' -> {content_type}")

def test_entity_extraction():
    """Test named entity extraction."""
    logger.info("\n=== Testing Entity Extraction ===")
    
    chunker = SemanticChunker()
    
    text = "The FDA approved Lisinopril for treating hypertension. Dr. Smith from Johns Hopkins University conducted the clinical trial."
    
    entities = chunker.extract_entities(text)
    logger.info(f"Text: {text}")
    logger.info(f"Entities: {entities}")

def test_rag_integration():
    """Test integration with RAG system."""
    logger.info("\n=== Testing RAG Integration ===")
    
    try:
        from rag import RAGSystem
        # Create RAG system with semantic chunking
        rag = RAGSystem(chunking_strategy="hybrid")
        logger.info("RAG system initialized with semantic chunking")
        
        # Test if it can handle the semantic chunker
        logger.info(f"Chunking strategy: {rag.chunking_strategy}")
        logger.info(f"Semantic chunker loaded: {rag.semantic_chunker is not None}")
        
    except Exception as e:
        logger.error(f"Error testing RAG integration: {e}")

if __name__ == "__main__":
    logger.info("Starting semantic chunking tests...")
    
    test_semantic_chunker()
    test_content_classification() 
    test_entity_extraction()
    test_rag_integration()
    
    logger.info("Tests completed!")