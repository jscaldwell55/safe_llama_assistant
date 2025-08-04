# conversational_agent.py

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from conversation import conversation_manager
from llm_client import call_answerability_agent, call_base_assistant
from prompt import format_answerability_prompt, format_conversational_prompt, ACKNOWLEDGE_GAP_PROMPT

logger = logging.getLogger(__name__)

class ConversationMode(Enum):
    GENERAL = "general"
    SESSION_END = "session_end"

@dataclass
class ConversationResponse:
    text: str
    mode: ConversationMode
    requires_generation: bool
    debug_info: Optional[Dict[str, Any]] = None
    # Add context to pass to the final guard
    context: Optional[str] = None

class ConversationalAgent:
    """
    Orchestrates the conversation by first checking if a query is answerable
    from RAG context before deciding how to generate a response.
    """
    def __init__(self):
        # Greetings can be handled without RAG or generation
        self.greetings = {"hello", "hi", "hey", "good morning", "good afternoon"}

    def _is_greeting(self, query: str) -> bool:
        return query.lower().strip() in self.greetings

    def process_conversation(self, query: str) -> ConversationResponse:
        """Processes the query using the new RAG-then-Check workflow."""
        if conversation_manager.should_end_session():
            return ConversationResponse(
                text="Session limit reached. Please start a new conversation.",
                mode=ConversationMode.SESSION_END,
                requires_generation=False,
                debug_info={"reason": "Session limit"}
            )

        if self._is_greeting(query):
            return ConversationResponse(
                text="Hello! How can I help you with our documentation today?",
                mode=ConversationMode.GENERAL,
                requires_generation=False,
                debug_info={"reason": "Handled as simple greeting."}
            )

        # 1. ALWAYS PERFORM RAG SEARCH
        enhanced_query = conversation_manager.get_enhanced_query(query)
        logger.info(f"Performing RAG search for query: {enhanced_query}")
        try:
            from rag import retrieve_and_format_context
            # Assuming a function that retrieves and formats context chunks into a single string
            context_str = retrieve_and_format_context(enhanced_query)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            context_str = ""

        # 2. PERFORM ANSWERABILITY CHECK (LLM-based gate)
        answerability_prompt = format_answerability_prompt(query, context_str)
        classification, rationale = call_answerability_agent(answerability_prompt)
        
        debug_info = {
            "enhanced_query": enhanced_query,
            "context_retrieved_length": len(context_str),
            "answerability_classification": classification,
            "answerability_rationale": rationale,
        }

        # 3. CONDITIONAL GENERATION
        if classification in ["FULLY_ANSWERABLE", "PARTIALLY_ANSWERABLE"]:
            logger.info("Query is answerable. Generating grounded response.")
            # This is where you would implement the 2-step (skeleton->synthesis) generation.
            # For simplicity here, we'll use a direct generation prompt.
            generation_prompt = format_conversational_prompt(
                query,
                context_str,
                conversation_manager.get_formatted_history()
            )
            response_text = call_base_assistant(generation_prompt)
        else: # NOT_ANSWERABLE
            logger.info("Query is not answerable from context. Generating gap acknowledgement.")
            # Use a specific prompt to generate a safe "I don't know" response
            prompt = ACKNOWLEDGE_GAP_PROMPT.format(user_question=query, rationale=rationale)
            response_text = call_base_assistant(prompt)

        return ConversationResponse(
            text=response_text,
            mode=ConversationMode.GENERAL,
            requires_generation=True,
            debug_info=debug_info,
            context=context_str  # Pass context for the final guard
        )

# Global instance
conversational_agent = ConversationalAgent()