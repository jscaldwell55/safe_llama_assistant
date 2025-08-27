# conversation.py - Simplified Conversation Management

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import WELCOME_MESSAGE, MAX_CONVERSATION_TURNS, SESSION_TIMEOUT_MINUTES

try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    st = None
    HAS_STREAMLIT = False

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConversationContext:
    """Conversation state"""
    turns: List[ConversationTurn] = field(default_factory=list)
    session_started: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    turn_count: int = 0

class ConversationManager:
    """
    Manages conversation state for Claude context
    Simplified version focused on maintaining conversation history
    """
    
    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS):
        self.max_turns = max_turns
        self.session_timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES) if SESSION_TIMEOUT_MINUTES > 0 else None
        
        # Initialize conversation state
        if HAS_STREAMLIT and st is not None:
            if "conversation_context" not in st.session_state:
                self._init_new_conversation()
                logger.info("Initialized new conversation in session state")
            else:
                self.conversation = st.session_state["conversation_context"]
                logger.debug(f"Restored conversation with {len(self.conversation.turns)} turns")
        else:
            self._init_new_conversation()
    
    def _init_new_conversation(self):
        """Initialize a new conversation"""
        self.conversation = ConversationContext()
        
        # Add welcome message if configured
        if WELCOME_MESSAGE:
            self.conversation.turns.append(
                ConversationTurn(role="assistant", content=WELCOME_MESSAGE)
            )
            logger.debug("Added welcome message to conversation")
        
        # Store in Streamlit session state
        if HAS_STREAMLIT and st is not None:
            st.session_state["conversation_context"] = self.conversation
        
        logger.info("Started new conversation")
    
    def start_new_conversation(self):
        """Explicitly start a new conversation"""
        self._init_new_conversation()
        logger.info("Conversation reset by user")
    
    def add_turn(self, role: str, content: str):
        """Add a turn to the conversation"""
        if not self.conversation:
            self._init_new_conversation()
        
        # Create new turn
        turn = ConversationTurn(role=role, content=content)
        self.conversation.turns.append(turn)
        self.conversation.turn_count += 1
        self.conversation.last_activity = datetime.now()
        
        # Trim conversation if exceeds max turns (keep recent ones)
        if self.max_turns > 0 and len(self.conversation.turns) > self.max_turns:
            # Keep welcome message if it exists, then most recent turns
            if self.conversation.turns[0].content == WELCOME_MESSAGE:
                self.conversation.turns = [self.conversation.turns[0]] + self.conversation.turns[-(self.max_turns-1):]
            else:
                self.conversation.turns = self.conversation.turns[-self.max_turns:]
            logger.debug(f"Trimmed conversation to {len(self.conversation.turns)} turns")
        
        # Update session state
        if HAS_STREAMLIT and st is not None:
            st.session_state["conversation_context"] = self.conversation
        
        logger.info(f"Added {role} turn (total turns: {self.conversation.turn_count})")
    
    def get_turns(self) -> List[Dict[str, str]]:
        """Get all conversation turns as list of dicts"""
        if not self.conversation:
            return []
        
        # Convert to format expected by Claude API
        turns = []
        for turn in self.conversation.turns:
            # Skip welcome message in history sent to Claude
            if turn.content == WELCOME_MESSAGE and turn.role == "assistant":
                continue
            turns.append({
                "role": turn.role,
                "content": turn.content
            })
        
        return turns
    
    def get_display_turns(self) -> List[Dict[str, str]]:
        """Get turns for display (includes welcome message)"""
        if not self.conversation:
            return []
        
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.conversation.turns
        ]
    
    def check_session_timeout(self) -> bool:
        """Check if session has timed out"""
        if not self.session_timeout or not self.conversation:
            return False
        
        time_since_activity = datetime.now() - self.conversation.last_activity
        is_timeout = time_since_activity > self.session_timeout
        
        if is_timeout:
            logger.warning(f"Session timeout detected ({time_since_activity.seconds}s since last activity)")
        
        return is_timeout
    
    def get_session_info(self) -> Dict[str, any]:
        """Get session information for logging"""
        if not self.conversation:
            return {"status": "no_conversation"}
        
        return {
            "turn_count": self.conversation.turn_count,
            "message_count": len(self.conversation.turns),
            "session_duration": str(datetime.now() - self.conversation.session_started),
            "last_activity": str(datetime.now() - self.conversation.last_activity) + " ago",
            "is_timeout": self.check_session_timeout()
        }

# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

def get_conversation_manager() -> ConversationManager:
    """Get singleton ConversationManager instance"""
    if HAS_STREAMLIT and st is not None:
        if "conversation_manager" not in st.session_state:
            st.session_state["conversation_manager"] = ConversationManager()
            logger.debug("Created new ConversationManager in session state")
        return st.session_state["conversation_manager"]
    
    # Fallback for non-Streamlit contexts
    global _cm_singleton
    try:
        _cm_singleton
    except NameError:
        _cm_singleton = ConversationManager()
        logger.debug("Created global ConversationManager singleton")
    return _cm_singleton