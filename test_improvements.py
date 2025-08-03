#!/usr/bin/env python3
"""
Test script for RAG improvements - focuses on data cleaning and prompt formatting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from context_formatter import ContextFormatter
from prompt import format_conversational_prompt

def test_data_cleaning():
    """Test the new data cleaning functionality"""
    print("🧹 Testing data cleaning functionality...")
    
    formatter = ContextFormatter()
    
    # Test cases with various FAQ-style formatting
    test_chunks = [
        "User Question: What are the side effects?\nAnswer: The main side effects include nausea and headache.",
        "Question: How should I take this medication?\nResponse: Take as directed by your physician.",
        "A: The dosage is 10mg daily.\nQ: When should I take it?",
        "This is clean pharmaceutical text about Lexapro (escitalopram) for depression treatment.",
        "1. Question: Is this safe?\n1. Answer: Yes, when used as directed."
    ]
    
    print("Original chunks:")
    for i, chunk in enumerate(test_chunks):
        print(f"  {i+1}: {chunk[:60]}...")
    
    print("\nCleaned chunks:")
    cleaned_chunks = []
    for i, chunk in enumerate(test_chunks):
        cleaned = formatter.clean_chunk_text(chunk)
        cleaned_chunks.append(cleaned)
        print(f"  {i+1}: {cleaned[:60]}...")
    
    # Check if cleaning worked
    success = True
    problem_patterns = ["User Question:", "Answer:", "Question:", "Response:", "A:", "Q:"]
    for cleaned in cleaned_chunks:
        for pattern in problem_patterns:
            if pattern in cleaned:
                print(f"❌ Found problematic pattern '{pattern}' in cleaned text")
                success = False
    
    if success:
        print("✅ Data cleaning successful - no FAQ patterns remain")
    else:
        print("❌ Data cleaning failed - some patterns still present")
    
    return success

def test_prompt_improvements():
    """Test the improved prompt template"""
    print("\n🎯 Testing prompt template improvements...")
    
    test_query = "What are the side effects of this medication?"
    test_context = "LEXAPRO may cause nausea, headache, and dizziness in some patients."
    
    # Format using the new conversational prompt
    formatted_prompt = format_conversational_prompt(
        query=test_query,
        formatted_context=test_context,
        conversation_context="",
        intent="question"
    )
    
    print("Generated prompt structure:")
    print("="*50)
    print(formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)
    print("="*50)
    
    # Check for key improvements
    improvements_found = []
    
    if "based *only* on the context provided" in formatted_prompt:
        improvements_found.append("Explicit context-only instruction")
    
    if "Do NOT copy the context verbatim" in formatted_prompt:
        improvements_found.append("Anti-verbatim copying instruction")
    
    if "---CONTEXT---" in formatted_prompt and "---END CONTEXT---" in formatted_prompt:
        improvements_found.append("Clear context boundaries")
    
    if "synthesize the information into a natural response" in formatted_prompt:
        improvements_found.append("Synthesis instruction")
    
    print(f"\nPrompt improvements found:")
    for improvement in improvements_found:
        print(f"  ✅ {improvement}")
    
    success = len(improvements_found) >= 3
    if success:
        print("✅ Prompt template successfully improved")
    else:
        print("❌ Prompt template missing key improvements")
    
    return success

def test_end_to_end_formatting():
    """Test the complete formatting pipeline"""
    print("\n🔄 Testing end-to-end formatting pipeline...")
    
    formatter = ContextFormatter()
    
    # Simulate messy retrieved chunks
    messy_chunks = [
        "User Question: What is Lexapro used for?\nAnswer: Lexapro is used to treat depression and anxiety.",
        "Question: What are common side effects?\nResponse: Common side effects include nausea, headache, and dizziness.",
        "Clean pharmaceutical information: Lexapro (escitalopram) is an SSRI antidepressant."
    ]
    
    test_query = "What is Lexapro and what are its side effects?"
    
    # Test the enhanced formatting
    formatted_context = formatter.format_enhanced_context(
        chunks=messy_chunks,
        query=test_query,
        conversation_context="",
        conversation_entities=[]
    )
    
    print("Formatted context preview:")
    print("-" * 30)
    print(formatted_context[:300] + "..." if len(formatted_context) > 300 else formatted_context)
    print("-" * 30)
    
    # Check that FAQ formatting was removed
    success = True
    problem_patterns = ["User Question:", "Answer:", "Question:", "Response:"]
    for pattern in problem_patterns:
        if pattern in formatted_context:
            print(f"❌ Found problematic pattern '{pattern}' in formatted context")
            success = False
    
    if success:
        print("✅ End-to-end formatting successful")
    else:
        print("❌ End-to-end formatting failed")
    
    return success

def main():
    """Run all improvement tests"""
    print("🚀 Testing RAG system improvements...\n")
    
    tests = [
        ("Data Cleaning", test_data_cleaning),
        ("Prompt Improvements", test_prompt_improvements),
        ("End-to-End Formatting", test_end_to_end_formatting),
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
    print("📊 Improvement Test Results:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All improvements working correctly!")
        print("\n📋 Summary of fixes implemented:")
        print("✅ Added data cleaning to remove FAQ-style formatting")
        print("✅ Strengthened prompt with explicit instructions")
        print("✅ Added clear context boundaries")
        print("✅ Instructed model to synthesize, not copy")
        print("✅ Added anti-verbatim copying instructions")
    else:
        print("⚠️  Some improvements need attention.")

if __name__ == "__main__":
    main()