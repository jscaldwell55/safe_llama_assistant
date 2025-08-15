# conversation.py

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from config import WELCOME_MESSAGE, MAX_CONVERSATION_TURNS, SESSION_TIMEOUT_MINUTES

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None
    HAS_STREAMLIT = False

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConversationContext:
    turns: List[ConversationTurn] = field(default_factory=list)
    active_entities: List[str] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)  # Track actual activity
    session_started: datetime = field(default_factory=datetime.now)

class ConversationManager:
    """Manages conversational state (memory) and context building."""
    def __init__(
        self,
        max_turns: Optional[int] = None,
        session_timeout_minutes: Optional[int] = None
    ):
        # Allow config-driven defaults; 0/None disables turn cap
        self.max_turns = (MAX_CONVERSATION_TURNS if max_turns is None else max_turns) or 0
        timeout = SESSION_TIMEOUT_MINUTES if session_timeout_minutes is None else session_timeout_minutes
        self.session_timeout_minutes = timeout
        self.session_timeout = timedelta(minutes=timeout) if timeout > 0 else None
        
        # Initialize conversation state from Streamlit session state if available
        if HAS_STREAMLIT and st is not None:
            if "conversation_context" not in st.session_state:
                # First time initialization
                self._init_new_conversation()
            else:
                # Restore existing conversation
                self.conversation = st.session_state["conversation_context"]
                # Don't check timeout on init - it causes false positives
                # Timeout should only be checked after actual user interaction
        else:
            # Non-Streamlit context
            self._init_new_conversation()

    def _init_new_conversation(self):
        """Initialize a new conversation and store in session state if available"""
        self.conversation = ConversationContext()
        if WELCOME_MESSAGE:
            self.conversation.turns.append(ConversationTurn(role="assistant", content=WELCOME_MESSAGE))
        
        # Store in Streamlit session state
        if HAS_STREAMLIT and st is not None:
            st.session_state["conversation_context"] = self.conversation
        
        logger.info("Started new conversation")

    def _check_timeout(self) -> bool:
        """Check if session has timed out based on last activity"""
        if not self.session_timeout or not self.conversation:
            return False
        
        time_since_activity = datetime.now() - self.conversation.last_activity
        return time_since_activity > self.session_timeout

    def start_new_conversation(self):
        """Explicitly start a new conversation"""
        self._init_new_conversation()

    def add_turn(self, role: str, content: str):
        if not self.conversation:
            self._init_new_conversation()
        
        # Update last activity time
        self.conversation.last_activity = datetime.now()
        
        # Add the turn
        self.conversation.turns.append(ConversationTurn(role=role, content=content))

        # Lightweight entity heuristic
        entities = self._extract_entities(content)
        for e in entities:
            if e not in self.conversation.active_entities:
                self.conversation.active_entities.append(e)
        self.conversation.active_entities = self.conversation.active_entities[-5:]

        # Update session state
        if HAS_STREAMLIT and st is not None:
            st.session_state["conversation_context"] = self.conversation

        logger.info(f"Added '{role}' turn.")

    def get_turns(self) -> List[Dict[str, str]]:
        if not self.conversation:
            return []
        return [{"role": t.role, "content": t.content} for t in self.conversation.turns]

    def _extract_entities(self, text: str) -> List[str]:
        return re.findall(r'\b[A-Z][a-z]{2,}\b', text)

    def should_end_session(self) -> bool:
        """Check if session should end due to turn limit or timeout"""
        if not self.conversation:
            return False
        
        # Check turn limit
        if self.max_turns and len(self.conversation.turns) >= self.max_turns:
            logger.warning("Session turn limit reached.")
            return True
        
        # Check timeout based on last activity (not last turn)
        if self.session_timeout:
            if self._check_timeout():
                logger.warning("Session timed out due to inactivity.")
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

# Streamlit-session-scoped instance
def get_conversation_manager():
    if HAS_STREAMLIT and st is not None:
        if "conversation_manager" not in st.session_state:
            st.session_state["conversation_manager"] = ConversationManager()
        return st.session_state["conversation_manager"]
    # Fallback for non-Streamlit contexts
    global _cm_singleton  # type: ignore
    try:
        _cm_singleton
    except NameError:
        _cm_singleton = ConversationManager()  # type: ignore
    return _cm_singleton  # type: ignore