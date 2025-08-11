import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Use lazy loading functions to avoid circular imports at startup
def get_conversation_manager():
    from conversation import get_conversation_manager
    return get_conversation_manager()


def get_rag_retriever():
    # This function now encapsulates how to get the formatted context string
    from rag import retrieve_and_format_context
    return retrieve_and_format_context

logger = logging.getLogger(__name__)

class ConversationMode(Enum):
    GENERAL = "general"
    SESSION_END = "session_end"

@dataclass
class AgentDecision:
    """The output of the agent's decision process. It's a set of instructions for app.py."""
    mode: ConversationMode = ConversationMode.GENERAL
    requires_generation: bool = False
    context_str: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

class ConversationalAgent:
    """
    Orchestrates the conversation by first checking if a query is answerable
    from RAG context before deciding how to generate a response.
    """
    def __init__(self):
        self.greetings = {"hello", "hi", "hey", "good morning", "good afternoon"}

    def _is_greeting(self, query: str) -> bool:
        """Checks for simple, standalone greetings."""
        return query.lower().strip() in self.greetings

    def process_query(self, query: str) -> AgentDecision:
        """Processes the query and returns instructions for the main app loop."""
        conversation_manager = get_conversation_manager()

        if conversation_manager.should_end_session():
            return AgentDecision(
                mode=ConversationMode.SESSION_END,
                debug_info={"reason": "Session limit reached."}
            )

        if self._is_greeting(query):
            # For a simple greeting, we don't need to generate a response.
            # We can return the final text directly.
            return AgentDecision(
                requires_generation=False,
                debug_info={"reason": "Handled as simple greeting."}
            )

        # 1. ALWAYS PERFORM RAG SEARCH for non-greetings
        enhanced_query = conversation_manager.get_enhanced_query(query)
        context_str = ""
        try:
            retrieve_and_format_context = get_rag_retriever()
            context_str = retrieve_and_format_context(enhanced_query)
            logger.info(f"Retrieved context of length {len(context_str)} for query.")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)
            # Proceed with an empty context

        debug_info = {
            "enhanced_query": enhanced_query,
            "context_retrieved_length": len(context_str),
        }

        # 2. Return instructions to app.py
        # The agent's decision is that generation is required, and here's the context to use.
        return AgentDecision(
            requires_generation=True,
            context_str=context_str,
            debug_info=debug_info
        )

# Use a lazy-loading function for the global instance
_agent_instance = None
def get_conversational_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
    return _agent_instance
