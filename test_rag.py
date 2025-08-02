#!/usr/bin/env python3
"""
Simple test script for RAG functionality
"""

from rag import retrieve, rag_system
from prompt import format_base_prompt
from guard import evaluate_response

def test_rag_retrieval():
    """Test basic RAG retrieval functionality"""
    print("🔍 Testing RAG retrieval...")
    
    test_queries = [
        "What is this document about?",
        "What are the side effects?",
        "How should this be used?",
        "What are the contraindications?"
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        chunks = retrieve(query, k=3)
        
        if chunks:
            print(f"✅ Retrieved {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"   Chunk {i+1}: {chunk[:100]}...")
        else:
            print("❌ No chunks retrieved")
    
    return True

def test_prompt_formatting():
    """Test prompt formatting functionality"""
    print("\n🔧 Testing prompt formatting...")
    
    test_query = "What are the main points?"
    test_chunks = ["This is test chunk 1.", "This is test chunk 2."]
    
    formatted_prompt = format_base_prompt(test_query, test_chunks)
    
    if "Context:" in formatted_prompt and "User Question:" in formatted_prompt:
        print("✅ Prompt formatting works correctly")
        return True
    else:
        print("❌ Prompt formatting failed")
        return False

def test_guard_functionality():
    """Test guard evaluation (without calling HF endpoint)"""
    print("\n🛡️ Testing guard functionality...")
    
    # Test with a simple response that should pass basic checks
    test_context = "This is test context about safety protocols."
    test_question = "What are the safety protocols?"
    test_response = "Based on the provided context, the safety protocols include following proper procedures."
    
    try:
        # Test the quick safety check first
        from guard import guard_agent
        quick_check = guard_agent.quick_safety_check(test_response)
        
        if quick_check:
            print("✅ Quick safety check passed")
        else:
            print("❌ Quick safety check failed")
        
        return quick_check
    except Exception as e:
        print(f"❌ Guard test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RAG system tests...\n")
    
    tests = [
        ("RAG Retrieval", test_rag_retrieval),
        ("Prompt Formatting", test_prompt_formatting),
        ("Guard Functionality", test_guard_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")

if __name__ == "__main__":
    main()