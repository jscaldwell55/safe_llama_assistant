# prompts.py - Simplified prompts for Claude-based system

# Note: The main system prompt is now in llm_client.py as SYSTEM_PROMPT
# This file maintains compatibility with legacy code

# ============================================================================
# FALLBACK MESSAGES
# ============================================================================

NO_CONTEXT_FALLBACK = "I'm sorry, I don't have any information on that. Can I assist you with something else? If this is an emergency or you need immediate medical care, please call 911"
GUARD_FALLBACK = "I'm sorry, I can't discuss that. Can we talk about something else? If this is an emergency or you need immediate medical care, please call 911"

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# These are kept for backward compatibility but not actively used
def format_synthesizer_prompt(query: str, context: str = "") -> str:
    """Legacy - now handled by Claude client"""
    return f"Context: {context}\n\nQuestion: {query}"

def format_conversational_prompt(query: str, **kwargs) -> str:
    """Legacy - now handled by Claude client"""
    return query

def format_validation_prompt(context: str, query: str, response: str) -> str:
    """Legacy - validation now only checks grounding score"""
    return f"Check if response is grounded in context"

# Legacy constants for compatibility
ENHANCED_BRIDGE_PROMPT = "Not used - Claude handles this"
BRIDGE_SYNTHESIZER_SIMPLE_PROMPT = "Not used - Claude handles this"
NO_CONTEXT_PROMPT = NO_CONTEXT_FALLBACK
CONVERSATIONAL_PROMPT = "Not used - Claude handles this"
ENHANCED_GUARD_VALIDATION_PROMPT = "Not used - simplified to grounding only"

# Stub out other legacy functions
def format_threat_detection_prompt(query: str) -> str:
    return "Not used"

def format_fact_extraction_prompt(topic: str, context: str) -> str:
    return "Not used"

def format_intent_classification_prompt(query: str) -> str:
    return "Not used"

def format_empathetic_prompt(emotional_context: str) -> str:
    return "Not used"

def format_navigator_prompt(query: str, context: str) -> str:
    return format_synthesizer_prompt(query, context)

def format_guard_prompt(context, question, answer, conversation_history=None):
    return format_validation_prompt(context, question, answer)

# Legacy constants
BASE_ASSISTANT_PROMPT = "Not used - Claude handles this"
ACKNOWLEDGE_GAP_PROMPT = NO_CONTEXT_FALLBACK