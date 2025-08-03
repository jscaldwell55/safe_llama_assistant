#!/usr/bin/env python3
"""
Test script for conversational boundaries and RAG-only information policy.
"""

import os
import logging
from conversational_agent import conversational_agent, ConversationMode
from conversation import conversation_manager
from guard import guard_agent

# Setup environment
os.environ["HF_TOKEN"] = "dummy_token"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_conversation_modes():
    """Test different conversation modes and their responses"""
    logger.info("=== Testing Conversation Modes ===")
    
    test_cases = [
        # Greetings
        ("Hello", ConversationMode.GREETING),
        ("Hi there", ConversationMode.GREETING),
        ("Good morning", ConversationMode.GREETING),
        
        # Help requests
        ("What can you do?", ConversationMode.HELP),
        ("Help me", ConversationMode.HELP),
        ("How do I use this?", ConversationMode.HELP),
        
        # Information requests
        ("What is hypertension?", ConversationMode.INFORMATION_REQUEST),
        ("Tell me about drug interactions", ConversationMode.INFORMATION_REQUEST),
        ("How does Lexapro work?", ConversationMode.INFORMATION_REQUEST),
        
        # Chitchat
        ("Thank you", ConversationMode.CHITCHAT),
        ("That's helpful", ConversationMode.CHITCHAT),
        ("Okay", ConversationMode.CHITCHAT),
    ]
    
    for query, expected_mode in test_cases:
        actual_mode = conversational_agent.classify_conversation_mode(query)
        status = "✅" if actual_mode == expected_mode else "❌"
        logger.info(f"{status} '{query}' -> Expected: {expected_mode.value}, Got: {actual_mode.value}")

def test_rag_only_validation():
    """Test RAG-only content validation"""
    logger.info("\n=== Testing RAG-Only Validation ===")
    
    # Test cases with different context scenarios
    test_cases = [
        # No context - should only allow fallback/conversational
        {
            "context": "",
            "response": "I'm sorry, I don't seem to have any information on that.",
            "should_pass": True,
            "description": "Standard fallback with no context"
        },
        {
            "context": "",
            "response": "Hello! I'm here to help you.",
            "should_pass": True,
            "description": "Greeting with no context"
        },
        {
            "context": "",
            "response": "Hypertension is high blood pressure that affects millions of people.",
            "should_pass": False,
            "description": "Factual claim without context"
        },
        {
            "context": "",
            "response": "Based on research, this medication is effective.",
            "should_pass": False,
            "description": "Research claim without context"
        },
        
        # With context - should allow grounded responses
        {
            "context": "Lexapro is an antidepressant medication used to treat depression and anxiety disorders.",
            "response": "Based on our documentation, Lexapro is an antidepressant used for depression and anxiety.",
            "should_pass": True,
            "description": "Response grounded in context"
        },
        {
            "context": "Side effects may include nausea and headache.",
            "response": "According to the information, common side effects include nausea and headache.",
            "should_pass": True,
            "description": "Side effects from context"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        is_valid, reasoning = guard_agent._validate_rag_only_response(
            test_case["context"], 
            test_case["response"], 
            "test question"
        )
        
        expected = test_case["should_pass"]
        status = "✅" if is_valid == expected else "❌"
        
        logger.info(f"{status} Test {i+1}: {test_case['description']}")
        logger.info(f"   Expected: {expected}, Got: {is_valid}")
        logger.info(f"   Reasoning: {reasoning}")
        logger.info("")

def test_conversational_flow():
    """Test end-to-end conversational flow"""
    logger.info("=== Testing Conversational Flow ===")
    
    # Start fresh conversation
    conversation_manager.start_new_conversation()
    
    test_queries = [
        "Hello",
        "What can you help me with?", 
        "Tell me about drug interactions",
        "What about side effects?",
        "Thank you"
    ]
    
    for query in test_queries:
        logger.info(f"\n--- Processing: '{query}' ---")
        
        # Process through conversational agent
        conv_response = conversational_agent.process_conversation(query)
        
        logger.info(f"Mode: {conv_response.mode.value}")
        logger.info(f"Has RAG content: {conv_response.has_rag_content}")
        logger.info(f"Confidence: {conv_response.confidence}")
        logger.info(f"Response: {conv_response.text[:100]}...")
        
        if conv_response.follow_up_suggestions:
            logger.info(f"Follow-up suggestions: {conv_response.follow_up_suggestions}")

def test_follow_up_detection():
    """Test follow-up question detection"""
    logger.info("\n=== Testing Follow-up Detection ===")
    
    # Set up conversation context
    conversation_manager.start_new_conversation()
    conversation_manager.add_turn(
        "What is Lexapro?", 
        "Lexapro is an antidepressant medication.", 
        ["sample context"], 
        "lexapro"
    )
    
    follow_up_queries = [
        "What about side effects?",
        "And dosage?", 
        "Tell me more about it",
        "What else should I know?",
        "How does it work?"
    ]
    
    for query in follow_up_queries:
        is_follow_up = conversation_manager.is_follow_up_question(query)
        enhanced_query = conversation_manager.get_enhanced_query(query)
        
        status = "✅" if is_follow_up else "❌"
        logger.info(f"{status} '{query}' -> Follow-up: {is_follow_up}")
        logger.info(f"   Enhanced: '{enhanced_query}'")

def test_social_patterns():
    """Test social/conversational pattern recognition"""
    logger.info("\n=== Testing Social Patterns ===")
    
    social_queries = [
        ("Thank you so much!", "thanks"),
        ("That's very helpful", "thanks"), 
        ("Yes, that makes sense", "affirmations"),
        ("Can you explain that?", "clarification_requests"),
        ("Please help me understand", "politeness"),
        ("Hello there", "greetings")
    ]
    
    for query, expected_pattern in social_queries:
        # Test pattern matching
        patterns_to_test = {
            'thanks': conversational_agent.social_patterns['thanks'],
            'affirmations': conversational_agent.social_patterns['affirmations'],
            'clarification_requests': conversational_agent.social_patterns['clarification_requests'],
            'politeness': conversational_agent.social_patterns['politeness'],
            'greetings': conversational_agent.social_patterns['greetings']
        }
        
        matches = conversational_agent._matches_patterns(query.lower(), patterns_to_test[expected_pattern])
        status = "✅" if matches else "❌"
        
        logger.info(f"{status} '{query}' -> Expected pattern: {expected_pattern}, Matches: {matches}")

if __name__ == "__main__":
    logger.info("Starting conversational boundaries tests...")
    
    test_conversation_modes()
    test_rag_only_validation()
    test_conversational_flow()
    test_follow_up_detection()
    test_social_patterns()
    
    logger.info("\n✅ All conversational boundary tests completed!")