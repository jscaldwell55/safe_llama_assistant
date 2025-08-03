#!/usr/bin/env python3
"""
Test script for session limits and auto-reset functionality.
"""

import os
import logging

# Setup environment
os.environ["HF_TOKEN"] = "dummy_token"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after environment setup
from conversation import conversation_manager

def test_session_limit_tracking():
    """Test session limit tracking functionality"""
    logger.info("=== Testing Session Limit Tracking ===")
    
    # Start fresh
    conversation_manager.start_new_conversation()
    
    # Test initial state
    assert conversation_manager.get_turns_remaining() == 6
    assert not conversation_manager.is_session_limit_reached()
    assert not conversation_manager.should_end_session()
    logger.info("✅ Initial state: 6 turns remaining, session not ended")
    
    # Add turns and check limits
    for i in range(1, 7):  # Add 6 turns
        conversation_manager.add_turn(
            f"Question {i}", 
            f"Response {i}", 
            [f"context {i}"], 
            f"topic_{i}"
        )
        
        turns_remaining = conversation_manager.get_turns_remaining()
        is_limit_reached = conversation_manager.is_session_limit_reached()
        should_end = conversation_manager.should_end_session()
        
        logger.info(f"Turn {i}: {turns_remaining} remaining, limit_reached={is_limit_reached}, should_end={should_end}")
        
        if i < 6:
            assert turns_remaining == 6 - i
            assert not is_limit_reached
            assert not should_end
        else:  # i == 6
            assert turns_remaining == 0
            assert is_limit_reached
            assert should_end
    
    logger.info("✅ Session limit tracking works correctly")

def test_session_auto_reset():
    """Test automatic session reset after limit"""
    logger.info("\n=== Testing Session Auto-Reset ===")
    
    # Set up a session at the limit
    conversation_manager.start_new_conversation()
    
    # Add 6 turns to reach limit
    for i in range(6):
        conversation_manager.add_turn(
            f"Question {i+1}", 
            f"Response {i+1}", 
            [f"context {i+1}"]
        )
    
    # Verify we're at the limit
    assert conversation_manager.should_end_session()
    logger.info("✅ Session at limit (6 turns)")
    
    # Simulate session reset (like what happens in app.py)
    turn_count_before = len(conversation_manager.conversation.turns)
    conversation_manager.start_new_conversation()
    turn_count_after = len(conversation_manager.conversation.turns) if conversation_manager.conversation else 0
    
    assert turn_count_before == 6
    assert turn_count_after == 0
    assert conversation_manager.get_turns_remaining() == 6
    assert not conversation_manager.should_end_session()
    
    logger.info("✅ Session auto-reset works correctly")

def test_conversation_context_with_limits():
    """Test conversation context building with session limits"""
    logger.info("\n=== Testing Conversation Context with Limits ===")
    
    conversation_manager.start_new_conversation()
    
    # Add several turns
    turns_data = [
        ("What is Lexapro?", "Lexapro is an antidepressant", ["lexapro context"], "lexapro"),
        ("What about side effects?", "Common side effects include nausea", ["side effects context"], "lexapro"),
        ("What's the dosage?", "Typical dose is 10-20mg daily", ["dosage context"], "lexapro"),
        ("Any interactions?", "May interact with MAOIs", ["interactions context"], "lexapro"),
        ("How long to work?", "Usually 4-6 weeks", ["timeline context"], "lexapro"),
    ]
    
    for i, (query, response, context, topic) in enumerate(turns_data):
        conversation_manager.add_turn(query, response, context, topic)
        
        # Test context retrieval
        context_str = conversation_manager.get_conversation_context()
        turns_remaining = conversation_manager.get_turns_remaining()
        
        logger.info(f"Turn {i+1}: {turns_remaining} remaining")
        logger.info(f"  Context length: {len(context_str)} chars")
        
        # Should have context after first turn
        if i > 0:
            assert len(context_str) > 0
            assert "lexapro" in context_str.lower()
        
        # Should not end session yet
        assert not conversation_manager.should_end_session()
    
    # Add one more turn to reach limit
    conversation_manager.add_turn("Thank you", "You're welcome", [], "lexapro")
    
    # Now should be at limit
    assert conversation_manager.should_end_session()
    assert conversation_manager.get_turns_remaining() == 0
    
    logger.info("✅ Conversation context works correctly with session limits")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    logger.info("\n=== Testing Edge Cases ===")
    
    # Test with no conversation
    conversation_manager.conversation = None
    assert conversation_manager.get_turns_remaining() == 6
    assert not conversation_manager.is_session_limit_reached()
    assert not conversation_manager.should_end_session()
    logger.info("✅ No conversation state handled correctly")
    
    # Test empty conversation
    conversation_manager.start_new_conversation()
    assert len(conversation_manager.conversation.turns) == 0
    assert conversation_manager.get_turns_remaining() == 6
    logger.info("✅ Empty conversation handled correctly")
    
    # Test exactly at limit
    for i in range(6):
        conversation_manager.add_turn(f"Q{i}", f"A{i}", [f"C{i}"])
    
    assert conversation_manager.is_session_limit_reached()
    assert conversation_manager.should_end_session()
    assert conversation_manager.get_turns_remaining() == 0
    logger.info("✅ Exact limit boundary handled correctly")

def test_conversation_flow_simulation():
    """Simulate a realistic conversation flow"""
    logger.info("\n=== Testing Realistic Conversation Flow ===")
    
    conversation_manager.start_new_conversation()
    
    # Simulate realistic conversation
    conversation_flow = [
        ("Hello", "greeting"),
        ("What can you help me with?", "help"),
        ("Tell me about Lexapro", "information_request"),
        ("What are the side effects?", "follow_up"),
        ("How about drug interactions?", "follow_up"),
        ("Thank you for the information", "chitchat"),
        # This 7th interaction should trigger session end
        ("Anything else?", "session_end_trigger")
    ]
    
    for i, (query, expected_type) in enumerate(conversation_flow):
        turn_number = i + 1
        turns_remaining_before = conversation_manager.get_turns_remaining()
        should_end_before = conversation_manager.should_end_session()
        
        logger.info(f"\n--- Turn {turn_number}: '{query}' ---")
        logger.info(f"Before: {turns_remaining_before} turns remaining, should_end={should_end_before}")
        
        if should_end_before:
            logger.info(f"🔴 Session should end before processing this turn")
            break
        else:
            # Simulate adding turn (would happen after processing in real app)
            conversation_manager.add_turn(query, f"Response to: {query}", [f"context_{i}"])
            
            turns_remaining_after = conversation_manager.get_turns_remaining()
            should_end_after = conversation_manager.should_end_session()
            
            logger.info(f"After: {turns_remaining_after} turns remaining, should_end={should_end_after}")
            
            if should_end_after:
                logger.info(f"🟡 Session limit reached after this turn")
    
    logger.info("✅ Realistic conversation flow simulation completed")

if __name__ == "__main__":
    logger.info("Starting session limits and auto-reset tests...\n")
    
    test_session_limit_tracking()
    test_session_auto_reset()
    test_conversation_context_with_limits()
    test_edge_cases()
    test_conversation_flow_simulation()
    
    logger.info("\n✅ All session limit tests completed successfully!")