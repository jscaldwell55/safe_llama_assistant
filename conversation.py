# conversation.py
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from config import MAX_CONVERSATION_TURNS, SESSION_TIMEOUT_MINUTES, WELCOME_MESSAGE

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
    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS * 2, session_timeout_minutes: int = SESSION_TIMEOUT_MINUTES):
        # max_turns counts individual turns; 2 per exchange
        self.max_turns = max_turns
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.conversation: Optional[ConversationContext] = None
        self.start_new_conversation()

    def start_new_conversation(self):
        self.conversation = ConversationContext()
        logger.info("Started new conversation")
        # Seed a welcome message that shows on app load and on "New Conversation"
        if WELCOME_MESSAGE:
            self.conversation.turns.append(ConversationTurn(role="assistant", content=WELCOME_MESSAGE))

    def add_turn(self, role: str, content: str):
        if not self.conversation:
            self.start_new_conversation()
        turn = ConversationTurn(role=role, content=content)
        self.conversation.turns.append(turn)

        entities = self._extract_entities(content)
        for entity in entities:
            if entity not in self.conversation.active_entities:
                self.conversation.active_entities.append(entity)
        self.conversation.active_entities = self.conversation.active_entities[-5:]

        logger.info(f"Added '{role}' turn. Total turns: {len(self.conversation.turns)}/{self.max_turns}")

    def get_turns(self) -> List[Dict[str, str]]:
        if not self.conversation:
            return []
        return [{"role": t.role, "content": t.content} for t in self.conversation.turns]

    def _extract_entities(self, text: str) -> List[str]:
        # Simple heuristic: capitalized words of len≥3
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
        recent_turns = self.conversation.turns[-4:]
        history = []
        for turn in recent_turns:
            role_formatted = "Human" if turn.role == "user" else "Assistant"
            history.append(f"{role_formatted}: {turn.content}")
        return "\n".join(history)

    def get_enhanced_query(self, original_query: str) -> str:
        query_lower = original_query.lower()
        is_follow_up = any(word in query_lower for word in ['it', 'that', 'they', 'them', 'more', 'also'])
        if is_follow_up and self.conversation and self.conversation.active_entities:
            context_entities = ", ".join(self.conversation.active_entities)
            enhanced_query = f"{original_query} (related to: {context_entities})"
            logger.info(f"Enhanced query for RAG: {enhanced_query}")
            return enhanced_query
        return original_query

# Global singleton
_conversation_manager_instance = None
def get_conversation_manager():
    global _conversation_manager_instance
    if _conversation_manager_instance is None:
        _conversation_manager_instance = ConversationManager()
    return _conversation_manager_instance
