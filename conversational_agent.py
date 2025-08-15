# conversational_agent.py
import logging
import re
from typing import Dict, Any
from dataclasses import dataclass, field
from enum import Enum

def get_conversation_manager():
    from conversation import get_conversation_manager
    return get_conversation_manager()

def get_rag_retriever():
    from rag import retrieve_and_format_context
    return retrieve_and_format_context

logger = logging.getLogger(__name__)

class ConversationMode(Enum):
    GENERAL = "general"
    SESSION_END = "session_end"

@dataclass
class AgentDecision:
    mode: ConversationMode = ConversationMode.GENERAL
    requires_generation: bool = False
    context_str: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

class ConversationalAgent:
    def __init__(self):
        # Word-level greetings and phrase-level greetings
        self.greeting_words = {"hi", "hello", "hey"}
        self.greeting_phrases = {"good morning", "good afternoon", "good evening"}

    def _is_greeting(self, query: str) -> bool:
        """
        Consider it a greeting ONLY if it is a short, standalone greeting,
        with no question mark and no additional content.
        Examples considered greeting: "hi", "hello", "hey", "good morning"
        Examples NOT considered greeting: "hello, can you...", "hi there could you...", "hello?"
        """
        q = query.lower().strip()

        # If it contains a question mark, treat as a real query
        if "?" in q:
            return False

        # Exact phrase greetings
        if q in self.greeting_phrases:
            return True

        # Word-only greetings with <= 3 tokens, all in greeting_words
        words = re.findall(r"[a-z]+", q)
        if 1 <= len(words) <= 3 and all(w in self.greeting_words for w in words):
            return True

        return False

    def process_query(self, query: str) -> AgentDecision:
        conversation_manager = get_conversation_manager()

        if conversation_manager.should_end_session():
            return AgentDecision(
                mode=ConversationMode.SESSION_END,
                debug_info={"reason": "Session limit reached."}
            )

        if self._is_greeting(query):
            return AgentDecision(
                requires_generation=False,
                debug_info={"reason": "Handled as simple greeting."}
            )

        enhanced_query = conversation_manager.get_enhanced_query(query)
        context_str = ""
        try:
            retrieve_and_format_context = get_rag_retriever()
            context_str = retrieve_and_format_context(enhanced_query)
            logger.info(f"Retrieved context of length {len(context_str)} for query.")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)

        debug_info = {
            "enhanced_query": enhanced_query,
            "context_retrieved_length": len(context_str),
        }

        return AgentDecision(
            requires_generation=True,
            context_str=context_str,
            debug_info=debug_info
        )

_agent_instance = None
def get_conversational_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
    return _agent_instance
