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
    follow_up_context: Optional[str] = None

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
    
    def __init__(self, max_turns: int = 6, session_timeout_minutes: int = 30):
        self.max_turns = max_turns
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
        
        # Update current topic and entities
        if topic:
            self.conversation.current_topic = topic
            
        # Extract entities (drugs, conditions) from query
        entities = self._extract_entities(query)
        for entity in entities:
            if entity not in self.conversation.active_entities:
                self.conversation.active_entities.append(entity)
        
        # Don't truncate turns here - we want to track all turns for session limit
        logger.info(f"Added conversation turn. Total turns: {len(self.conversation.turns)}/{self.max_turns}")
    
    def _extract_entities(self, text: str) -> List[str]:
        """Basic entity extraction for drugs and medical terms"""
        text_lower = text.lower()
        entities = []
        
        # Common drug names and medical terms
        medical_terms = [
            'lexapro', 'escitalopram', 'ssri', 'antidepressant',
            'depression', 'anxiety', 'side effects', 'dosage',
            'withdrawal', 'medication', 'prescription'
        ]
        
        for term in medical_terms:
            if term in text_lower:
                entities.append(term)
                
        return entities
    
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
        """Check if session should end (reached limit)"""
        return self.is_session_limit_reached()
    
    def get_conversation_context(self) -> str:
        """Build conversation context for the prompt"""
        if not self.conversation or not self.conversation.turns:
            return ""
            
        # Check if session has timed out
        if self.conversation.turns:
            last_turn = self.conversation.turns[-1]
            if datetime.now() - last_turn.timestamp > self.session_timeout:
                logger.info("Conversation session timed out, starting fresh")
                self.start_new_conversation()
                return ""
        
        context_parts = []
        
        # Add current topic if available
        if self.conversation.current_topic:
            context_parts.append(f"Current topic: {self.conversation.current_topic}")
            
        # Add active entities
        if self.conversation.active_entities:
            entities_str = ", ".join(self.conversation.active_entities[-5:])  # Last 5 entities
            context_parts.append(f"Active discussion about: {entities_str}")
        
        # Add recent conversation history (last 3 turns)
        recent_turns = self.conversation.turns[-3:]
        if len(recent_turns) > 1:  # Only add if there's actual history
            history_parts = []
            for turn in recent_turns[:-1]:  # Exclude current turn
                history_parts.append(f"Q: {turn.query[:100]}...")
                history_parts.append(f"A: {turn.response[:150]}...")
            
            if history_parts:
                context_parts.append("Recent conversation:")
                context_parts.extend(history_parts)
        
        return "\n".join(context_parts) if context_parts else ""
    
    def is_follow_up_question(self, query: str) -> bool:
        """Determine if this is a follow-up to previous conversation"""
        if not self.conversation or not self.conversation.turns:
            return False
            
        query_lower = query.lower().strip()
        
        # Check for follow-up indicators
        follow_up_patterns = [
            'what about', 'and', 'also', 'too', 'as well',
            'can you tell me more', 'more about', 'additional',
            'other', 'else', 'further', 'continue', 'next'
        ]
        
        # Check for pronouns referring to previous context
        pronouns = ['it', 'that', 'this', 'they', 'them', 'those', 'these']
        
        # Check for questions that start with follow-up words
        starts_with_followup = any(query_lower.startswith(pattern) for pattern in follow_up_patterns)
        contains_pronouns = any(pronoun in query_lower for pronoun in pronouns)
        
        # Short questions are often follow-ups
        is_short_question = len(query.split()) <= 4 and '?' in query
        
        return starts_with_followup or contains_pronouns or is_short_question
    
    def get_enhanced_query(self, original_query: str) -> str:
        """Enhance query with conversation context for better retrieval"""
        if not self.is_follow_up_question(original_query):
            return original_query
            
        enhanced_parts = [original_query]
        
        # Add current topic context
        if self.conversation and self.conversation.current_topic:
            enhanced_parts.append(f"regarding {self.conversation.current_topic}")
            
        # Add active entities for context
        if self.conversation and self.conversation.active_entities:
            recent_entities = self.conversation.active_entities[-3:]
            enhanced_parts.extend(recent_entities)
        
        enhanced_query = " ".join(enhanced_parts)
        logger.info(f"Enhanced query: '{original_query}' -> '{enhanced_query}'")
        
        return enhanced_query
    
    def classify_intent(self, query: str) -> Tuple[str, Optional[str]]:
        """Enhanced intent classification with topic detection"""
        query_lower = query.lower().strip()
        
        # Greeting detection
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(greeting in query_lower for greeting in greetings):
            return "greeting", None
            
        # Help detection  
        if any(word in query_lower for word in ["help", "what can you do", "capabilities"]):
            return "help_request", None
            
        # Follow-up detection
        if self.is_follow_up_question(query):
            topic = self.conversation.current_topic if self.conversation else None
            return "follow_up", topic
            
        # Topic detection for medical queries
        if any(term in query_lower for term in ['side effects', 'effects', 'reactions']):
            return "medical_question", "side_effects"
        elif any(term in query_lower for term in ['dosage', 'dose', 'how much', 'take']):
            return "medical_question", "dosage"  
        elif any(term in query_lower for term in ['lexapro', 'escitalopram']):
            return "medical_question", "lexapro"
        elif any(term in query_lower for term in ['withdrawal', 'stopping', 'discontinue']):
            return "medical_question", "withdrawal"
        else:
            return "medical_question", "general"

# Global conversation manager
conversation_manager = ConversationManager()