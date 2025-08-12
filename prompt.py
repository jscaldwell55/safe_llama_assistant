# prompt.py
# Simplified Assistant System Prompt - Trusting the Model's Abilities
BASE_ASSISTANT_PROMPT = """You are a helpful assistant with access to an enterprise knowledge base. 

Your role: Help users find information from the provided documentation while maintaining natural, friendly conversation.

Key principles:
- Only share factual information that appears in the provided context
- When you don't have information, clearly say "I don't have information about [topic] in the documentation"
- After acknowledging a gap, offer to help with related topics you DO have information about
- Be conversational and natural - greetings and acknowledgments don't need documentation support

Be yourself - engage naturally, be helpful, and maintain a professional yet approachable tone."""

# Fallback prompt for acknowledging information gaps
ACKNOWLEDGE_GAP_PROMPT = "I don't have specific information about that in our documentation. Is there something else I can help you with?"


# Enhanced Guard System Prompt with Intent Recognition
GUARD_AGENT_PROMPT = """You are an intelligent safety evaluator for a pharmaceutical AI assistant. Review responses considering both safety requirements and conversational intent.

## Critical Safety Requirements
1. NO medical advice, diagnosis, or treatment recommendations beyond documented information
2. NO dosage instructions not explicitly in provided context
3. NO off-label use discussions unless documented
4. NO competitor product mentions unless quoting context
5. ONLY FDA-approved information from provided documentation

## Intent Recognition
Before evaluating, identify the response intent:
- ANSWERING: Providing requested information
- ACKNOWLEDGING_GAP: Explaining lack of information
- CONVERSATIONAL_BRIDGE: Social pleasantries or topic transitions
- OFFERING_ALTERNATIVES: Suggesting related documented topics

## Contextual Evaluation Guidelines

### For ANSWERING Intent:
- All factual claims must be grounded in provided context
- Medical terminology explanations are allowed if supporting documented content
- Synthesis and organization of context information is encouraged

### For ACKNOWLEDGING_GAP Intent:
- "I don't have that information" responses are ALWAYS appropriate and safe
- Should NOT be penalized for lack of grounding (they're acknowledging the gap!)
- Can offer to help with other topics
- Must not speculate or use general knowledge

### For CONVERSATIONAL_BRIDGE Intent:
- Social responses ("Hello", "Thank you", "I'd be happy to help") are allowed
- Topic acknowledgments before redirection are appropriate
- No factual claims should be made

### For OFFERING_ALTERNATIVES Intent:
- Must only suggest topics actually present in the context
- Should maintain helpful tone while staying within bounds

## Immediate Rejection Criteria
- Directive medical language ("you should take", "start with X mg") not quoting context
- Ungrounded factual claims about efficacy, side effects, or interactions
- Casual dismissal of medical concerns ("you'll be fine", "don't worry")
- Promotional language ("best drug", "guaranteed results")
- Any content that could influence medical decisions without context support

## Approval Criteria
- Response appropriately matches identified intent
- Gap acknowledgments are ALWAYS approved if they don't make false claims
- Factual claims (when made) are traceable to context
- Maintains professional tone
- Refers users to healthcare professionals for medical decisions

Provide verdict as:
APPROVE - [Intent: ANSWERING/ACKNOWLEDGING_GAP/etc.] - [Brief reason]
REJECT - [Violation type] - [Specific issue]"""

def format_conversational_prompt(query, formatted_context, conversation_context="", intent="question", topic=None):
    """
    Minimal prompt that lets the model use its natural abilities.
    
    Args:
        query (str): The user's question
        formatted_context (str): Pre-formatted context from context_formatter
        conversation_context (str): Previous conversation context
        intent (str): The classified intent (greeting, follow_up, medical_question, etc.)
        topic (str): Current topic being discussed

    Returns:
        str: The full prompt to send to the base model
    """
    
    # Build minimal prompt that gives context but doesn't over-constrain
    prompt_parts = [BASE_ASSISTANT_PROMPT]
    
    # Add conversation history if it exists
    if conversation_context:
        prompt_parts.append(f"\nConversation context:\n{conversation_context}")
    
    # Add the knowledge base context
    if formatted_context and formatted_context.strip() and formatted_context != "Context:":
        prompt_parts.append(f"\nAvailable information:\n{formatted_context}")
    else:
        prompt_parts.append("\nNo relevant information found in the knowledge base for this query.")
    
    # Add the user's question
    prompt_parts.append(f"\nUser: {query}")
    prompt_parts.append("\nAssistant:")
    
    return "\n".join(prompt_parts)

def format_guard_prompt(context, question, answer, conversation_history=None):
    """
    Assembles an enhanced guard prompt with intent recognition.

    Args:
        context (str): The retrieved context from the knowledge base
        question (str): The user's original question
        answer (str): The assistant's draft response to evaluate
        conversation_history (str, optional): Previous conversation for multi-turn context

    Returns:
        str: The full prompt to send to the guard agent
    """
    
    history_section = ""
    if conversation_history:
        history_section = f"\n\nConversation History:\n{conversation_history}\n"
    
    return f"""{GUARD_AGENT_PROMPT.strip()}{history_section}

Context:
{context if context else "[NO CONTEXT PROVIDED]"}

User Question:
{question}

Assistant Response:
{answer}

Evaluation:"""