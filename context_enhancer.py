# context_enhancer.py - Enhance queries with conversation context

import logging
import re
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class QueryContextEnhancer:
    """
    Enhances queries by adding context from conversation history
    to handle pronouns and references
    """
    
    def __init__(self):
        # Pronouns and references that indicate follow-up questions
        self.reference_indicators = [
            "it", "they", "them", "these", "those", "this", "that",
            "the ones", "the one", "ones", "same", "such",
            "above", "previous", "mentioned", "listed"
        ]
        
        # Topic keywords to track from previous messages
        self.medical_topics = [
            "side effect", "dosage", "dose", "interaction", "contraindication",
            "warning", "precaution", "administration", "storage", "pregnancy",
            "adverse reaction", "symptom", "treatment"
        ]
    
    def needs_context_enhancement(self, query: str) -> bool:
        """Check if query needs context from previous conversation"""
        query_lower = query.lower()
        
        # Check for references without clear subject
        has_reference = any(ref in query_lower for ref in self.reference_indicators)
        
        # Check if query is too short or vague
        is_vague = len(query.split()) < 5 and not any(topic in query_lower for topic in self.medical_topics)
        
        # Check for follow-up question patterns
        follow_up_patterns = [
            r"^what about",
            r"^how about", 
            r"^and ",
            r"^also",
            r"^but what",
            r"^what.*(?:most|worst|severe|serious|common)",
            r"which ones",
            r"any others?"
        ]
        
        is_follow_up = any(re.search(pattern, query_lower) for pattern in follow_up_patterns)
        
        return has_reference or is_vague or is_follow_up
    
    def extract_topic_from_history(self, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        """Extract the most recent relevant topic from conversation history"""
        if not conversation_history:
            return None
        
        # Look at last 3 messages (go backwards)
        for msg in reversed(conversation_history[-3:]):
            content = msg.get("content", "").lower()
            
            # Find medical topics mentioned
            for topic in self.medical_topics:
                if topic in content:
                    logger.info(f"Found recent topic in history: {topic}")
                    return topic
        
        return None
    
    def enhance_query(self, query: str, conversation_history: List[Dict[str, str]]) -> Tuple[str, bool]:
        """
        Enhance query with context from conversation history
        Returns (enhanced_query, was_enhanced)
        """
        # Check if enhancement is needed
        if not self.needs_context_enhancement(query):
            return query, False
        
        # Extract topic from history
        topic = self.extract_topic_from_history(conversation_history)
        
        if not topic:
            logger.debug("No clear topic found in recent history")
            return query, False
        
        # Enhance the query
        query_lower = query.lower()
        
        # Different enhancement strategies based on query pattern
        if "most severe" in query_lower or "worst" in query_lower or "serious" in query_lower:
            if "side effect" in topic:
                enhanced = f"{query} (asking about severe side effects of Journvax)"
            else:
                enhanced = f"{query} (asking about severe {topic} of Journvax)"
        
        elif "which ones" in query_lower or "what are they" in query_lower:
            enhanced = f"{query} (referring to {topic} of Journvax)"
        
        elif query_lower.startswith("what about"):
            enhanced = f"{query} (asking about {topic})"
        
        else:
            # Generic enhancement
            enhanced = f"{query} (in context of {topic} for Journvax)"
        
        logger.info(f"Enhanced query: '{query}' -> '{enhanced}'")
        return enhanced, True
    
    def generate_clarification_for_vague_query(self, query: str, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Generate a clarification question for vague queries
        """
        topic = self.extract_topic_from_history(conversation_history)
        
        if topic and "side effect" in topic:
            return (
                "I see you're asking a follow-up question about side effects. "
                "To help me provide the most relevant information:\n\n"
                "Are you asking about:\n"
                "1. The most severe/serious side effects?\n"
                "2. The most common side effects?\n"
                "3. Specific side effects you're concerned about?\n"
                "4. How to manage side effects?\n\n"
                "Please let me know, or feel free to rephrase your question."
            )
        
        return None

# Singleton instance
_enhancer_instance: Optional[QueryContextEnhancer] = None

def get_query_enhancer() -> QueryContextEnhancer:
    """Get singleton query enhancer"""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = QueryContextEnhancer()
    return _enhancer_instance