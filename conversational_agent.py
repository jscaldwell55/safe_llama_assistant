import logging
import re
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# --- Lazy singletons (cached to avoid repeated imports) ---
_cm = None
_retriever: Optional[Callable[[str], str]] = None

def _get_conversation_manager():
    global _cm
    if _cm is None:
        from conversation import get_conversation_manager
        _cm = get_conversation_manager()
    return _cm

def _get_rag_retriever():
    global _retriever
    if _retriever is None:
        from rag import retrieve_and_format_context
        _retriever = retrieve_and_format_context
    return _retriever

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
        if not query:
            return False

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
        cm = _get_conversation_manager()

        if cm.should_end_session():
            return AgentDecision(
                mode=ConversationMode.SESSION_END,
                debug_info={"reason": "Session limit reached."}
            )

        # Simple greetings are handled directly by the UI
        if self._is_greeting(query or ""):
            return AgentDecision(
                requires_generation=False,
                debug_info={"reason": "Handled as simple greeting."}
            )

        # Guard against empty/whitespace input
        q_norm = (query or "").strip()
        if not q_norm:
            return AgentDecision(
                requires_generation=False,
                debug_info={"reason": "Empty query."}
            )

        # Enhance the query with recent entities to improve retrieval
        enhanced_query = cm.get_enhanced_query(q_norm)
        logger.info(f"Enhanced query for RAG: {enhanced_query}")

        # Retrieve + format context (RAG)
        context_str = ""
        try:
            retriever = _get_rag_retriever()
            context_str = retriever(enhanced_query)  # returns a compact, formatted context block
            logger.info(f"Retrieved context of length {len(context_str)} for query.")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)

        debug_info = {
            "enhanced_query": enhanced_query,
            "context_retrieved_length": len(context_str),
        }

        # Keep contract: we still generate even if context is empty.
        # The system prompt will force a gap acknowledgment; Guard enforces grounding/safety.
        return AgentDecision(
            requires_generation=True,
            context_str=context_str,
            debug_info=debug_info
        )

_agent_instance: Optional[ConversationalAgent] = None
def get_conversational_agent() -> ConversationalAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
    return _agent_instance
