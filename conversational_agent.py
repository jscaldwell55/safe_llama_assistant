import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from conversation import conversation_manager
import re

logger = logging.getLogger(__name__)

class ConversationMode(Enum):
    """Different modes of conversation engagement"""
    GENERAL = "general"  # Let the model handle most conversations naturally
    SESSION_END = "session_end"

@dataclass
class ConversationResponse:
    """Structure for conversation responses"""
    text: str
    mode: ConversationMode
    has_rag_content: bool
    requires_generation: bool  # Whether we need the LLM to generate
    confidence: float
    follow_up_suggestions: Optional[List[str]] = None
    debug_info: Optional[Dict[str, Any]] = None

class ConversationalAgent:
    """
    Simplified conversational agent that trusts the model's natural abilities.
    
    Key principles:
    1. Let the model engage naturally in conversation
    2. Only intervene for session management
    3. Trust the guard to ensure safety
    4. Maintain conversation context
    """
    
    def __init__(self):
        # We only need to handle session end - everything else goes to the model
        pass
    
    def _query_needs_rag(self, query: str) -> bool:
        """
        Determine if a query likely needs RAG content.
        Simple heuristic to avoid loading RAG for greetings, simple questions, etc.
        """
        query_lower = query.lower().strip()
        
        # Skip RAG for common greetings and simple interactions
        skip_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)',
            r'^(thanks|thank you|bye|goodbye|see you)',
            r'^(yes|no|ok|okay|sure|got it)',
            r'^(how are you|what\'s up|wassup)',
            r'^(who are you|what are you)',
            r'^(can you help|help me)',
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, query_lower):
                logger.info(f"Skipping RAG for conversational query: {query[:50]}")
                return False
        
        # Skip for very short queries (likely conversational)
        if len(query_lower.split()) <= 2:
            logger.info(f"Skipping RAG for short query: {query}")
            return False
        
        # Default to using RAG for substantive queries
        return True
    
    def process_conversation(self, query: str) -> ConversationResponse:
        """Simplified conversation processing that trusts the model"""
        
        # Only special handling for session limits
        if conversation_manager.should_end_session():
            return ConversationResponse(
                text="We've reached the conversation limit for this session. Thank you for chatting! Please feel free to start a new conversation to continue exploring our knowledge base.",
                mode=ConversationMode.SESSION_END,
                has_rag_content=False,
                requires_generation=False,
                confidence=1.0,
                follow_up_suggestions=["Click 'New Conversation' to start fresh"]
            )
        
        # For everything else, let the model handle it naturally
        # Check if query needs RAG content
        if self._query_needs_rag(query):
            # Only load RAG when actually needed
            enhanced_query = conversation_manager.get_enhanced_query(query)
            logger.info(f"Enhanced query for RAG: {enhanced_query}")
            
            try:
                # Lazy import RAG retrieve function
                from rag import retrieve
                rag_content = retrieve(enhanced_query)
                has_content = rag_content and len(rag_content) > 0
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
                rag_content = []
                has_content = False
        else:
            # Skip RAG entirely for simple queries
            enhanced_query = query
            rag_content = []
            has_content = False
            logger.info("Skipping RAG retrieval for this query")
        
        # Return response indicating we need generation
        return ConversationResponse(
            text="",  # Will be filled by LLM
            mode=ConversationMode.GENERAL,
            has_rag_content=has_content,
            requires_generation=True,
            confidence=0.9 if has_content else 0.5,
            debug_info={
                "enhanced_query": enhanced_query,
                "rag_chunks_found": len(rag_content) if rag_content else 0
            }
        )

# Global conversational agent instance
conversational_agent = ConversationalAgent()