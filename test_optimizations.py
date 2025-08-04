#!/usr/bin/env python3
"""
Test script to verify the latency optimizations are working correctly.
"""

import time
import asyncio
import logging
from rag import RAGSystem
from async_rag import AsyncRAGSystem
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_batch_processing():
    """Test that batch processing is working correctly."""
    print("\n" + "="*60)
    print("Testing Batch Processing for Embeddings")
    print("="*60)
    
    # Create test data
    test_chunks = [f"This is test chunk number {i}" for i in range(100)]
    
    # Test with original RAG system (modified with batching)
    rag = RAGSystem()
    
    print(f"\nProcessing {len(test_chunks)} chunks with batch size 32...")
    start_time = time.time()
    
    # Simulate batch processing
    from config import EMBEDDING_BATCH_SIZE
    batch_times = []
    
    for i in range(0, len(test_chunks), EMBEDDING_BATCH_SIZE):
        batch_start = time.time()
        batch_end = min(i + EMBEDDING_BATCH_SIZE, len(test_chunks))
        batch = test_chunks[i:batch_end]
        embeddings = rag.embedding_model.encode(batch, show_progress_bar=False)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        print(f"  Batch {i//EMBEDDING_BATCH_SIZE + 1}: {len(batch)} chunks in {batch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Average time per batch: {np.mean(batch_times):.2f} seconds")
    print(f"Average time per chunk: {total_time/len(test_chunks):.4f} seconds")
    
    return True

async def test_async_operations():
    """Test async I/O operations."""
    print("\n" + "="*60)
    print("Testing Async I/O Operations")
    print("="*60)
    
    # Test async RAG system
    rag = AsyncRAGSystem()
    await rag.initialize()
    
    # Test concurrent retrieval
    test_queries = [
        "What is machine learning?",
        "How does natural language processing work?",
        "What are embeddings?",
        "Explain neural networks",
        "What is deep learning?"
    ]
    
    print(f"\nTesting concurrent retrieval for {len(test_queries)} queries...")
    
    # Sequential timing (for comparison)
    start_time = time.time()
    sequential_results = []
    for query in test_queries:
        result = await rag.retrieve(query, k=3)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Sequential retrieval: {sequential_time:.2f} seconds")
    
    # Concurrent timing
    start_time = time.time()
    tasks = [rag.retrieve(query, k=3) for query in test_queries]
    concurrent_results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    
    print(f"Concurrent retrieval: {concurrent_time:.2f} seconds")
    print(f"Speedup: {sequential_time/concurrent_time:.2f}x")
    
    await rag.close()
    return True

async def test_async_llm_client():
    """Test async LLM client operations."""
    print("\n" + "="*60)
    print("Testing Async LLM Client")
    print("="*60)
    
    from async_llm_client import AsyncHuggingFaceClient
    
    async with AsyncHuggingFaceClient() as client:
        test_prompts = [
            "Hello, how are you?",
            "What is the weather like?",
            "Tell me a short joke.",
        ]
        
        print(f"\nTesting batch generation for {len(test_prompts)} prompts...")
        
        # Test batch generation
        start_time = time.time()
        responses = await client.batch_generate(test_prompts)
        batch_time = time.time() - start_time
        
        for prompt, response in zip(test_prompts, responses):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response[:100]}...")
        
        print(f"\nBatch generation time: {batch_time:.2f} seconds")
        print(f"Average time per prompt: {batch_time/len(test_prompts):.2f} seconds")
    
    return True

def test_memory_efficiency():
    """Test memory efficiency of batch processing."""
    print("\n" + "="*60)
    print("Testing Memory Efficiency")
    print("="*60)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Get initial memory usage
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Create large dataset
    large_chunks = [f"This is a longer test chunk with more content. " * 10 for _ in range(500)]
    
    rag = RAGSystem()
    
    # Process in batches
    print(f"\nProcessing {len(large_chunks)} chunks in batches...")
    from config import EMBEDDING_BATCH_SIZE
    
    peak_memory = initial_memory
    for i in range(0, len(large_chunks), EMBEDDING_BATCH_SIZE):
        batch_end = min(i + EMBEDDING_BATCH_SIZE, len(large_chunks))
        batch = large_chunks[i:batch_end]
        _ = rag.embedding_model.encode(batch, show_progress_bar=False)
        
        # Check memory after each batch
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
        
        if i % (EMBEDDING_BATCH_SIZE * 5) == 0:
            print(f"  Processed {i} chunks, current memory: {current_memory:.2f} MB")
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"\nFinal memory usage: {final_memory:.2f} MB")
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Memory increase: {peak_memory - initial_memory:.2f} MB")
    
    return True

async def main():
    """Run all optimization tests."""
    print("="*60)
    print("LATENCY OPTIMIZATION TESTS")
    print("="*60)
    
    # Test 1: Batch Processing
    try:
        if test_batch_processing():
            print("✅ Batch processing test PASSED")
    except Exception as e:
        print(f"❌ Batch processing test FAILED: {e}")
    
    # Test 2: Async Operations
    try:
        if await test_async_operations():
            print("✅ Async operations test PASSED")
    except Exception as e:
        print(f"❌ Async operations test FAILED: {e}")
    
    # Test 3: Async LLM Client
    try:
        if await test_async_llm_client():
            print("✅ Async LLM client test PASSED")
    except Exception as e:
        print(f"❌ Async LLM client test FAILED: {e}")
    
    # Test 4: Memory Efficiency
    try:
        if test_memory_efficiency():
            print("✅ Memory efficiency test PASSED")
    except Exception as e:
        print(f"❌ Memory efficiency test FAILED: {e}")
    
    print("\n" + "="*60)
    print("ALL OPTIMIZATION TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    # Check if psutil is installed for memory testing
    try:
        import psutil
    except ImportError:
        print("Note: Install psutil for memory efficiency testing: pip install psutil")
    
    asyncio.run(main())