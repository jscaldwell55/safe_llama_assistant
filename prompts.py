# prompts.py - Simplified Version for Bridge Synthesizer Only

# ============================================================================
# MAIN BRIDGE SYNTHESIZER PROMPT - SIMPLIFIED
# ============================================================================

BRIDGE_SYNTHESIZER_SIMPLE_PROMPT = """You are a helpful pharmaceutical assistant providing information about Journvax.

Using the documentation provided, give a natural, conversational response to the user's question.

IMPORTANT RULES:
- Present information in natural, flowing sentences
- NO bullet points, lists, or formatting marks
- NO headers like "Extracted Information" or "Based on documentation"
- Be accurate but conversational
- If information isn't in the documentation, say so politely
- Keep responses concise but complete

Available Documentation:
{context}

User Question: {query}

Response:"""

# ============================================================================
# ALTERNATIVE PROMPTS FOR DIFFERENT SCENARIOS
# ============================================================================

# When no context is available
NO_CONTEXT_PROMPT = """You are a helpful pharmaceutical assistant. 

The user has asked a question but no specific documentation is available.
Provide a helpful response that acknowledges this limitation.

User Question: {query}

Response:"""

# For conversational responses
CONVERSATIONAL_PROMPT = """You are a friendly pharmaceutical assistant focused on Journvax information.

Respond naturally and helpfully to the user's message.

User: {query}
Assistant:"""

# ============================================================================
# VALIDATION PROMPT (SIMPLIFIED)
# ============================================================================

GUARD_VALIDATION_PROMPT = """You are a safety validator for pharmaceutical responses.

Check if the response is:
1. Factually accurate based on the context provided
2. Safe and appropriate
3. Not providing dangerous medical advice

Output Format:
{
  "verdict": "APPROVE|REJECT",
  "safety_status": "SAFE|UNSAFE",
  "issues": ["..."],
  "confidence": 0.0-1.0
}

Context: {context}
User Query: {query}
Response: {response}

Validation:"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_synthesizer_prompt(query: str, context: str = "") -> str:
    """Format prompt for bridge synthesizer"""
    if context and context.strip():
        prompt = BRIDGE_SYNTHESIZER_SIMPLE_PROMPT
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{context}", context)
        return prompt
    else:
        # No context available
        return NO_CONTEXT_PROMPT.replace("{query}", query)

def format_conversational_prompt(query: str, **kwargs) -> str:
    """Format prompt for conversational responses"""
    return CONVERSATIONAL_PROMPT.replace("{query}", query)

def format_validation_prompt(context: str, query: str, response: str) -> str:
    """Format prompt for response validation"""
    prompt = GUARD_VALIDATION_PROMPT
    prompt = prompt.replace("{context}", context)
    prompt = prompt.replace("{query}", query)
    prompt = prompt.replace("{response}", response)
    return prompt

# ============================================================================
# LEGACY SUPPORT
# ============================================================================

# Keep these for backward compatibility
def format_intent_classification_prompt(query: str) -> str:
    """Legacy - not used in simplified version"""
    return f"Classify intent: {query}"

def format_empathetic_prompt(emotional_context: str) -> str:
    """Legacy - not used in simplified version"""
    return f"Provide support: {emotional_context}"

def format_navigator_prompt(query: str, context: str) -> str:
    """Legacy - routes to synthesizer"""
    return format_synthesizer_prompt(query, context)

def format_guard_prompt(context, question, answer, conversation_history=None):
    """Legacy guard prompt"""
    return format_validation_prompt(context, question, answer)

# Legacy constants
BASE_ASSISTANT_PROMPT = BRIDGE_SYNTHESIZER_SIMPLE_PROMPT
ACKNOWLEDGE_GAP_PROMPT = "I don't have specific information about that in our documentation. Is there something else I can help you with regarding Journvax?"

# Stub out legacy persona prompts
INTENT_CLASSIFIER_PROMPT = "Not used in simplified version"
EMPATHETIC_COMPANION_PROMPT = "Not used in simplified version"
INFORMATION_NAVIGATOR_PROMPT = "Not used in simplified version"