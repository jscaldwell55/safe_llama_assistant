#!/usr/bin/env python3
"""
Example script demonstrating the async optimizations for the RAG system.
This shows how to use the async conversational agent for better performance.
"""

import asyncio
import time
import logging
from async_conversational_agent import AsyncConversationalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_single_query():
    """Test a single query with the async agent."""
    async with AsyncConversationalAgent() as agent:
        query = "What is the main purpose of this system?"
        
        start_time = time.time()
        response = await agent.process_query(query)
        elapsed = time.time() - start_time
        
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        print(f"Time taken: {elapsed:.2f} seconds")

async def test_batch_queries():
    """Test multiple queries processed concurrently."""
    queries = [
        "What is the main purpose of this system?",
        "How does the RAG system work?",
        "What safety features are included?",
        "Can you explain the embedding process?",
        "What is semantic chunking?"
    ]
    
    async with AsyncConversationalAgent() as agent:
        print("\n=== Testing Batch Query Processing ===")
        
        # Process queries concurrently
        start_time = time.time()
        responses = await agent.process_batch_queries(queries)
        elapsed = time.time() - start_time
        
        for query, response in zip(queries, responses):
            print(f"\nQuery: {query}")
            print(f"Response: {response[:200]}...")  # Show first 200 chars
        
        print(f"\nTotal time for {len(queries)} queries: {elapsed:.2f} seconds")
        print(f"Average time per query: {elapsed/len(queries):.2f} seconds")

async def test_conversation_flow():
    """Test a conversation flow with multiple turns."""
    async with AsyncConversationalAgent() as agent:
        print("\n=== Testing Conversation Flow ===")
        
        conversation = [
            "Hello, can you help me understand this system?",
            "What kind of documents can it process?",
            "How does it ensure safety?",
            "Can you give me more details about the embedding model?",
            "Thanks for the information!"
        ]
        
        for turn, query in enumerate(conversation, 1):
            print(f"\n[Turn {turn}] User: {query}")
            
            start_time = time.time()
            response = await agent.process_query(query)
            elapsed = time.time() - start_time
            
            print(f"[Turn {turn}] Assistant: {response}")
            print(f"Response time: {elapsed:.2f} seconds")

async def benchmark_sync_vs_async():
    """Compare performance between sync and async operations."""
    print("\n=== Benchmarking Sync vs Async ===")
    
    # Test async version
    async with AsyncConversationalAgent() as agent:
        queries = ["What is RAG?"] * 5
        
        # Async concurrent processing
        start_time = time.time()
        await agent.process_batch_queries(queries)
        async_time = time.time() - start_time
        
        print(f"Async (concurrent) time for 5 queries: {async_time:.2f} seconds")
        
        # Sequential processing (for comparison)
        start_time = time.time()
        for query in queries:
            await agent.process_query(query)
        sequential_time = time.time() - start_time
        
        print(f"Sequential time for 5 queries: {sequential_time:.2f} seconds")
        print(f"Speedup: {sequential_time/async_time:.2f}x")

async def test_index_building():
    """Test async index building."""
    print("\n=== Testing Async Index Building ===")
    
    async with AsyncConversationalAgent() as agent:
        start_time = time.time()
        await agent.rebuild_index()
        elapsed = time.time() - start_time
        
        print(f"Index rebuilt in {elapsed:.2f} seconds")

async def main():
    """Run all tests."""
    print("=" * 60)
    print("Async RAG System Performance Tests")
    print("=" * 60)
    
    # Run individual tests
    await test_single_query()
    await test_batch_queries()
    await test_conversation_flow()
    await benchmark_sync_vs_async()
    
    # Optionally test index building (commented out by default as it takes time)
    # await test_index_building()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())