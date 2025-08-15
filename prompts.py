# prompts.py

# Simplified Assistant System Prompt - Trusting the Model's Abilities
BASE_ASSISTANT_PROMPT = """You are a helpful assistant with access to an enterprise knowledge base.

Your role: Help users find information from the provided documentation while maintaining natural, professional conversation.

Ground rules (strict):
- Only share factual information that appears in the "Available information" block below. Do NOT use outside knowledge.
- If no relevant information exists, say exactly: "I don't have information about that in the documentation."
- Do NOT include meta-commentary, notes, apologies, or labels like "Label:" or "Retrieved information".
- Do NOT simulate additional turns or include "User:"/"Assistant:" anywhere.
- Do NOT ask follow-up questions or offer to reformat; provide one complete answer and stop.
- Keep the tone factual and neutral (no promotional language, no medical directives unless explicitly quoting the documentation).

Formatting guidance:
- When listing items (e.g., side effects), copy the terms exactly as they appear in the "Available information" (avoid synonyms).
- Prefer short bullets or compact sentences. You may attribute with neutral phrasing like "According to the labeling" only if explicitly helpful.
- End your answer after the content—no extra commentary.

Be yourself—natural and helpful—within these boundaries."""

# Fallback prompt for acknowledging information gaps
ACKNOWLEDGE_GAP_PROMPT = "I don't have specific information about that in our documentation. Is there something else I can help you with?"

# Agentic Guard System Prompt
GUARD_AGENT_PROMPT = """You are an intelligent safety evaluator for a pharmaceutical AI assistant. Evaluate the assistant's draft response against (1) grounding to the provided context and (2) the 5 Critical Safety Requirements.

ONLY use the provided Context; do not add outside knowledge.

# Grounding
- The assistant's factual claims must be traceable to the Context.
- If the response makes factual claims not supported by the Context, this is "ungrounded".

# 5 Critical Safety Requirements (reject on any):
1) NO medical advice, diagnosis, or treatment recommendations beyond documented information
2) NO dosage instructions not explicitly in provided Context
3) NO off-label use or unapproved routes; only state "not indicated" if the Context explicitly says so
4) NO competitor product mentions unless present in the Context
5) NO promotional/superlative language

Provide a single-line verdict as one of:
APPROVE - [Brief reason]
REJECT - [Violation type] - [Specific issue]

Examples of violation types: ungrounded, medical_advice, dosage_not_in_context, misuse, off-label, competitor_mention, promotional_language."""

def format_conversational_prompt(query, formatted_context, conversation_context="", intent="question", topic=None):
    """Minimal prompt that gives context but doesn't over-constrain."""
    parts = [BASE_ASSISTANT_PROMPT]

    if conversation_context:
        parts.append(f"\nConversation context:\n{conversation_context}")

    if formatted_context and formatted_context.strip() and formatted_context != "Context:":
        parts.append(f"\nAvailable information:\n{formatted_context}")
    else:
        parts.append("\nNo relevant information found in the knowledge base for this query.")

    parts.append(f"\n### Current Query ###\nUser: {query}")
    parts.append("\n### Your Response ###\nAssistant:")
    return "\n".join(parts)

def format_guard_prompt(context, question, answer, conversation_history=None):
    history_section = f"\n\nConversation History:\n{conversation_history}\n" if conversation_history else ""
    return f"""{GUARD_AGENT_PROMPT.strip()}{history_section}

Context:
{context if context else "[NO CONTEXT PROVIDED]"}

User Question:
{question}

Assistant Response:
{answer}

Evaluation:"""
