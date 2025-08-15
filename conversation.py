# conversation.py
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from config import WELCOME_MESSAGE

# Streamlit is optional here so tests/CLI don't break
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # Allows non-Streamlit contexts

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConversationContext:
    turns: List[ConversationTurn] = field(default_factory=list)
    active_entities: List[str] = field(default_factory=list)

class ConversationManager:
    """Manages conversational state (memory) and context building."""
    def __init__(self, max_turns: int = 20, session_timeout_minutes: int = 30):
        # A turn is one user query + one assistant response → max_turns=20 allows 10 exchanges.
        self.max_turns = max_turns
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.conversation: Optional[ConversationContext] = None
        self.start_new_conversation()  # Fresh conversation on construction

    def start_new_conversation(self):
        """Starts a fresh conversation and seeds a welcome message."""
        self.conversation = ConversationContext()
        if WELCOME_MESSAGE:
            self.conversation.turns.append(
                ConversationTurn(role="assistant", content=WELCOME_MESSAGE)
            )
        logger.info("Started new conversation")

    def add_turn(self, role: str, content: str):
        if not self.conversation:
            self.start_new_conversation()

        turn = ConversationTurn(role=role, content=content)
        self.conversation.turns.append(turn)

        # Lightweight entity heuristic
        entities = self._extract_entities(content)
        for entity in entities:
            if entity not in self.conversation.active_entities:
                self.conversation.active_entities.append(entity)
        self.conversation.active_entities = self.conversation.active_entities[-5:]  # keep last 5

        logger.info(f"Added '{role}' turn. Total turns: {len(self.conversation.turns)}/{self.max_turns}")

    def get_turns(self) -> List[Dict[str, str]]:
        if not self.conversation:
            return []
        return [{"role": t.role, "content": t.content} for t in self.conversation.turns]

    def _extract_entities(self, text: str) -> List[str]:
        return re.findall(r'\b[A-Z][a-z]{2,}\b', text)

    def should_end_session(self) -> bool:
        if not self.conversation:
            return False
        if len(self.conversation.turns) >= self.max_turns:
            logger.warning("Session turn limit reached.")
            return True
        if self.conversation.turns:
            last_turn_time = self.conversation.turns[-1].timestamp
            if datetime.now() - last_turn_time > self.session_timeout:
                logger.warning("Session timed out.")
                return True
        return False

    def get_formatted_history(self) -> str:
        if not self.conversation or not self.conversation.turns:
            return ""
        recent = self.conversation.turns[-4:]  # last 2 user + 2 assistant turns
        history = []
        for turn in recent:
            role_fmt = "Human" if turn.role == "user" else "Assistant"
            history.append(f"{role_fmt}: {turn.content}")
        return "\n".join(history)

    def get_enhanced_query(self, original_query: str) -> str:
        ql = original_query.lower()
        is_follow_up = any(w in ql for w in ['it', 'that', 'they', 'them', 'more', 'also'])
        if is_follow_up and self.conversation and self.conversation.active_entities:
            context_entities = ", ".join(self.conversation.active_entities)
            enhanced = f"{original_query} (related to: {context_entities})"
            logger.info(f"Enhanced query for RAG: {enhanced}")
            return enhanced
        return original_query

# --- Streamlit-session-scoped instance ---
# On a browser refresh, Streamlit creates a NEW session_state → new manager → welcome message shown.
def get_conversation_manager():
    # Prefer Streamlit session-scoped instance (resets on browser refresh)
    if st is not None:
        if "conversation_manager" not in st.session_state:
            st.session_state["conversation_manager"] = ConversationManager()
        return st.session_state["conversation_manager"]

    # Fallback for non-Streamlit contexts (tests, scripts)
    # Single-process singleton (fine for unit tests)
    global _cm_singleton  # type: ignore
    try:
        _cm_singleton
    except NameError:
        _cm_singleton = ConversationManager()  # type: ignore
    return _cm_singleton  # type: ignore
