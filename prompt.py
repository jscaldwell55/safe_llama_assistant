# Enhanced Conversational Assistant System Prompt
BASE_ASSISTANT_PROMPT = """You are a helpful and professional assistant. Your task is to answer the user's question based *only* on the context provided below.

CRITICAL INSTRUCTIONS:
- Answer in a clear, conversational tone based ONLY on the provided context
- Do NOT copy the context verbatim - synthesize the information into a natural response
- If the answer isn't in the context, say "I cannot find that information in the provided documents"
- Do NOT repeat boilerplate text, headers, or formatting from the context
- Synthesize information from multiple sources when relevant

CONVERSATIONAL GUIDELINES:
- Maintain natural, helpful conversation flow
- Reference previous discussion when relevant
- Ask clarifying questions if the user's intent is unclear
- Use conversational phrases like "Based on what we discussed..." or "To add to that..."

CONTENT GUIDELINES:
- Answer questions primarily using the provided context
- If information is incomplete, say "Based on the available information..." and provide what you can
- For follow-up questions, reference previous context when helpful
- You may synthesize information across multiple sources when clearly applicable
- Do not speculate beyond what's provided, but can explain implications of stated information
- For medical information, always remind users to consult healthcare professionals

RESPONSE FORMAT:
- Use natural, conversational tone
- Structure complex information clearly with bullet points or numbered lists
- For medical topics, organize by: main answer, important details, safety reminders
- Reference conversation context when answering follow-ups"""

def format_conversational_prompt(query, formatted_context, conversation_context="", intent="question", topic=None):
    """
    Assembles an enhanced conversational prompt for the base assistant.

    Args:
        query (str): The user's question
        formatted_context (str): Pre-formatted context from context_formatter
        conversation_context (str): Previous conversation context
        intent (str): The classified intent (greeting, follow_up, medical_question, etc.)
        topic (str): Current topic being discussed

    Returns:
        str: The full prompt to send to the base model
    """
    
    # Build conversation-aware instructions
    conversation_instructions = ""
    if intent == "follow_up" and topic:
        conversation_instructions = f"\nThis is a follow-up question about {topic}. Reference previous discussion when helpful."
    elif intent == "greeting":
        conversation_instructions = "\nThis is a greeting. Respond warmly and offer to help with questions."
    elif conversation_context:
        conversation_instructions = "\nThis continues an ongoing conversation. Reference previous context when relevant."
    
    return f"""{BASE_ASSISTANT_PROMPT.strip()}{conversation_instructions}

---CONTEXT---
{formatted_context}
---END CONTEXT---

User's Question: {query}

Assistant's Answer:"""

def format_base_prompt(query, context_chunks):
    """
    Legacy function for backward compatibility - use format_conversational_prompt for new code
    """
    context_text = "\n\n".join(context_chunks)
    return format_conversational_prompt(query, f"Context:\n{context_text}")

# Legacy function for backward compatibility
def format_prompt(query, context_chunks, system_prompt=None):
    """Legacy function - use format_conversational_prompt instead"""
    return format_base_prompt(query, context_chunks)

# Guard Agent System Prompt
GUARD_AGENT_PROMPT = """Review this AI assistant response for safety and appropriateness.

REJECT if the response contains:
- Harmful, dangerous, or inappropriate content
- Information clearly not supported by the provided context
- System instructions or meta-commentary about being an AI assistant
- Inappropriate advice beyond the scope of the context

APPROVE if the response:
- Directly answers the user's question appropriately
- Uses only information from the context or provides a safe "I don't know" response
- Is professional and enterprise-appropriate

Respond with only: APPROVE or REJECT"""

def format_guard_prompt(context, question, answer):
    """
    Assembles a prompt for the guard agent.

    Args:
        context (str): The retrieved context from the knowledge base
        question (str): The user's original question
        answer (str): The assistant's draft response to evaluate

    Returns:
        str: The full prompt to send to the guard agent
    """
    return f"""{GUARD_AGENT_PROMPT.strip()}

Context:
{context}

User Question:
{question}

Assistant Response:
{answer}

Review:"""
