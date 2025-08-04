import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    query: str
    response: str
    context_used: List[str]
    timestamp: datetime
    topic: Optional[str] = None

@dataclass
class ConversationContext:
    """Maintains conversation state and context"""
    turns: List[ConversationTurn]
    current_topic: Optional[str] = None
    active_entities: List[str] = None  # Track mentioned drugs, conditions, etc.
    
    def __post_init__(self):
        if self.active_entities is None:
            self.active_entities = []

class ConversationManager:
    """Manages conversational state and context building"""
    
    def __init__(self, max_turns: int = 10, session_timeout_minutes: int = 30):
        self.max_turns = max_turns  # Changed to reasonable default
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.conversation: Optional[ConversationContext] = None
        
    def start_new_conversation(self):
        """Start a fresh conversation"""
        self.conversation = ConversationContext(turns=[])
        logger.info("Started new conversation")
    
    def add_turn(self, query: str, response: str, context_used: List[str], topic: str = None):
        """Add a conversation turn"""
        if not self.conversation:
            self.start_new_conversation()
            
        turn = ConversationTurn(
            query=query,
            response=response, 
            context_used=context_used,
            timestamp=datetime.now(),
            topic=topic
        )
        
        self.conversation.turns.append(turn)
        
        # Update current topic if provided
        if topic:
            self.conversation.current_topic = topic
            
        # Extract entities from both query and response for better tracking
        entities = self._extract_entities(query + " " + response)
        for entity in entities:
            if entity not in self.conversation.active_entities:
                self.conversation.active_entities.append(entity)
        
        # Keep only recent entities to avoid clutter
        if len(self.conversation.active_entities) > 10:
            self.conversation.active_entities = self.conversation.active_entities[-10:]
        
        logger.info(f"Added turn {len(self.conversation.turns)}/{self.max_turns}")
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities - can be extended with NER if needed"""
        text_lower = text.lower()
        entities = []
        
        # This is a simple keyword approach - could be replaced with proper NER
        # Extract any capitalized multi-word phrases (likely drug/condition names)
        import re
        
        # Find capitalized words/phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend([e.lower() for e in capitalized if len(e) > 3])
        
        # Also look for specific medical patterns
        medical_patterns = [
            r'\b\d+\s*mg\b',  # Dosages
            r'\b(?:tablet|capsule|pill)s?\b',  # Forms
            r'\b(?:daily|twice|three times)\b',  # Frequency
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates
    
    def is_session_limit_reached(self) -> bool:
        """Check if the conversation has reached the turn limit"""
        if not self.conversation:
            return False
        return len(self.conversation.turns) >= self.max_turns
    
    def get_turns_remaining(self) -> int:
        """Get number of turns remaining in current session"""
        if not self.conversation:
            return self.max_turns
        return max(0, self.max_turns - len(self.conversation.turns))
    
    def should_end_session(self) -> bool:
        """Check if session should end (reached limit or timeout)"""
        if self.is_session_limit_reached():
            return True
            
        # Check for timeout
        if self.conversation and self.conversation.turns:
            last_turn = self.conversation.turns[-1]
            if datetime.now() - last_turn.timestamp > self.session_timeout:
                logger.info("Session timed out")
                return True
                
        return False
    
    def get_conversation_context(self) -> str:
        """Build minimal conversation context that helps the model"""
        if not self.conversation or not self.conversation.turns:
            return ""
        
        # Even simpler - just the recent exchanges
        recent_turns = self.conversation.turns[-2:]  # Last 2 turns only
        if len(recent_turns) <= 1:
            return ""
        
        # Natural format
        history = []
        for turn in recent_turns[:-1]:  # Exclude current turn
            history.append(f"Human: {turn.query}")
            history.append(f"Assistant: {turn.response[:200]}")  # Truncate long responses
        
        return "\n".join(history)
    
    def get_enhanced_query(self, original_query: str) -> str:
        """Enhance query for better RAG retrieval"""
        # Simple approach: if query seems like a follow-up, add context
        if not self.conversation or not self.conversation.turns:
            return original_query
        
        query_lower = original_query.lower().strip()
        
        # Check for pronouns or references that need context
        needs_context = any(word in query_lower.split() for word in [
            'it', 'that', 'this', 'they', 'them', 'those', 'these',
            'more', 'also', 'other', 'another', 'else'
        ])
        
        if not needs_context:
            return original_query
        
        # Add recent topic/entity context
        context_parts = [original_query]
        
        # Add recent entities for context
        if self.conversation.active_entities:
            # Add most relevant recent entities
            recent_entities = self.conversation.active_entities[-3:]
            if recent_entities:
                context_parts.append(f"(context: {', '.join(recent_entities)})")
        
        enhanced = " ".join(context_parts)
        if enhanced != original_query:
            logger.info(f"Enhanced query: '{original_query}' -> '{enhanced}'")
        
        return enhanced
    
    def classify_intent(self, query: str) -> Tuple[str, Optional[str]]:
        """Simple intent classification - let the model handle most nuances"""
        query_lower = query.lower().strip()
        
        # Only classify obvious cases
        if any(greeting in query_lower.split() for greeting in ["hello", "hi", "hey"]):
            return "greeting", None
        
        # Everything else is just a question/statement
        # Extract topic if it's obvious
        topic = None
        if "side effect" in query_lower:
            topic = "side_effects"
        elif "dosage" in query_lower or "dose" in query_lower:
            topic = "dosage"
        elif "withdrawal" in query_lower or "stopping" in query_lower:
            topic = "withdrawal"
        
        return "general", topic

# Global conversation manager with reasonable defaults
conversation_manager = ConversationManager(max_turns=10)