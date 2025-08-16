# off_topic_handler.py - Graceful handling of off-topic but harmless requests

"""
Handles off-topic requests gracefully without inappropriately pivoting to medication.
Provides helpful alternatives when possible.
"""

import re
from typing import Optional, Tuple
from enum import Enum

class OffTopicType(Enum):
    """Types of off-topic requests"""
    CREATIVE_CONTENT = "creative_content"  # Stories, poems, songs
    GENERAL_ADVICE = "general_advice"  # Non-medical life advice
    TECHNICAL_HELP = "technical_help"  # Computer, math, coding help
    CONVERSATIONAL = "conversational"  # Casual chat
    JOURNVAX_RELATED = "journvax_related"  # Actually on-topic
    UNKNOWN = "unknown"  # Can't categorize

class OffTopicHandler:
    """
    Handles off-topic requests with appropriate, helpful responses.
    Avoids awkward pivots to medication topics.
    """
    
    def __init__(self):
        # Patterns for different off-topic categories
        self.creative_patterns = [
            'story', 'poem', 'song', 'tale', 'narrative', 'fiction',
            'bedtime story', 'fairy tale', 'write me', 'create a'
        ]
        
        self.general_advice_patterns = [
            'relationship', 'career', 'life advice', 'motivation',
            'bedtime routine', 'parenting', 'sleep tips', 'study tips'
        ]
        
        self.technical_patterns = [
            'code', 'programming', 'math', 'calculate', 'computer',
            'debug', 'algorithm', 'formula', 'equation'
        ]
        
        self.conversational_patterns = [
            'how are you', 'what do you think', 'your opinion',
            'tell me about yourself', 'favorite', 'prefer'
        ]
        
        self.journvax_keywords = [
            'journvax', 'medication', 'prescription', 'side effect',
            'dosage', 'dose', 'pharmaceutical', 'drug interaction'
        ]
        
        # Helpful alternative responses for each category
        self.responses = {
            OffTopicType.CREATIVE_CONTENT: (
                "I'm not able to create stories or creative content, as I'm specifically designed "
                "to provide information about Journvax. However, there are many wonderful resources "
                "for bedtime stories online, or I'd be happy to share some tips on creating "
                "calming bedtime routines if that would be helpful."
            ),
            OffTopicType.GENERAL_ADVICE: (
                "While I'm specifically designed to provide Journvax information rather than "
                "general advice, I understand this is important to you. For topics like this, "
                "you might find helpful resources from specialists in that area. "
                "Is there anything about Journvax I can help you with instead?"
            ),
            OffTopicType.TECHNICAL_HELP: (
                "I'm a pharmaceutical information assistant focused on Journvax, "
                "so I'm not equipped to help with technical or mathematical questions. "
                "For those topics, you might want to consult specialized resources. "
                "Is there anything about Journvax I can assist you with?"
            ),
            OffTopicType.CONVERSATIONAL: (
                "I appreciate the friendly question! I'm an assistant specifically designed "
                "to provide information about Journvax. How can I help you with "
                "pharmaceutical information today?"
            ),
            OffTopicType.UNKNOWN: (
                "I'm specifically designed to provide information about Journvax. "
                "I'm not able to help with that particular topic, but I'd be happy to answer "
                "any questions you have about Journvax, its usage, side effects, or safety information."
            )
        }
        
        # Special handling for certain requests
        self.special_cases = {
            'bedtime': self._handle_bedtime_request,
            'routine': self._handle_routine_request,
            'child': self._handle_child_request,
        }
    
    def categorize_request(self, query: str) -> OffTopicType:
        """
        Categorize the type of request.
        Returns: OffTopicType
        """
        query_lower = query.lower()
        
        # Check if it's actually about Journvax
        if any(keyword in query_lower for keyword in self.journvax_keywords):
            return OffTopicType.JOURNVAX_RELATED
        
        # Check for creative content
        if any(pattern in query_lower for pattern in self.creative_patterns):
            return OffTopicType.CREATIVE_CONTENT
        
        # Check for general advice
        if any(pattern in query_lower for pattern in self.general_advice_patterns):
            return OffTopicType.GENERAL_ADVICE
        
        # Check for technical help
        if any(pattern in query_lower for pattern in self.technical_patterns):
            return OffTopicType.TECHNICAL_HELP
        
        # Check for conversational
        if any(pattern in query_lower for pattern in self.conversational_patterns):
            return OffTopicType.CONVERSATIONAL
        
        return OffTopicType.UNKNOWN
    
    def generate_response(self, query: str) -> Optional[str]:
        """
        Generate appropriate response for off-topic request.
        Returns: Response string or None if on-topic
        """
        category = self.categorize_request(query)
        
        # If it's on-topic, return None to continue normal processing
        if category == OffTopicType.JOURNVAX_RELATED:
            return None
        
        query_lower = query.lower()
        
        # Check for special cases that need custom responses
        for keyword, handler in self.special_cases.items():
            if keyword in query_lower:
                special_response = handler(query)
                if special_response:
                    return special_response
        
        # Return the appropriate category response
        return self.responses.get(category, self.responses[OffTopicType.UNKNOWN])
    
    def _handle_bedtime_request(self, query: str) -> Optional[str]:
        """Special handling for bedtime-related requests"""
        if 'story' in query.lower():
            return (
                "I'm not able to tell bedtime stories, but I can share some tips for "
                "creating a calming bedtime routine. Many parents find that keeping "
                "a consistent schedule — like reading a favorite book, dimming the lights, "
                "or playing gentle music — helps children settle down more easily. "
                "If you'd like specific health-related sleep tips, I'd recommend "
                "consulting with your pediatrician."
            )
        return None
    
    def _handle_routine_request(self, query: str) -> Optional[str]:
        """Special handling for routine-related requests"""
        if 'bedtime' in query.lower() or 'sleep' in query.lower():
            return (
                "While I can't provide general parenting advice, I can share that "
                "consistent routines are important for good rest. This includes "
                "regular sleep and wake times, and avoiding screens before bed. "
                "If you have questions about how medications like Journvax might "
                "affect sleep patterns, I'd be happy to help with that information."
            )
        return None
    
    def _handle_child_request(self, query: str) -> Optional[str]:
        """Special handling for child-related non-medical requests"""
        query_lower = query.lower()
        
        # Only handle if it's NOT about medication
        medical_terms = ['medication', 'medicine', 'dose', 'prescription', 'journvax']
        if any(term in query_lower for term in medical_terms):
            return None  # Let medical safety handle this
        
        if 'story' in query_lower or 'bedtime' in query_lower:
            return self._handle_bedtime_request(query)
        
        return None
    
    def should_handle(self, query: str) -> bool:
        """
        Check if this handler should process the query.
        Returns: True if off-topic, False if on-topic
        """
        category = self.categorize_request(query)
        return category != OffTopicType.JOURNVAX_RELATED

# ============================================================================
# INTEGRATION WITH GUARD
# ============================================================================

def integrate_with_guard(guard_instance):
    """
    Integrate off-topic handling with the guard system.
    Should be called during guard initialization.
    """
    handler = OffTopicHandler()
    
    # Store original validate_query
    original_validate_query = guard_instance.validate_query
    
    async def enhanced_validate_query(query: str):
        # First check if it's off-topic but harmless
        off_topic_response = handler.generate_response(query)
        
        if off_topic_response:
            from guard import ValidationDecision, ValidationResult, ThreatType
            
            return ValidationDecision(
                result=ValidationResult.REDIRECT,
                final_response=off_topic_response,
                reasoning="Off-topic request handled gracefully",
                confidence=0.95,
                threat_type=ThreatType.OFF_TOPIC,
                should_log=False  # Don't log harmless off-topic requests
            )
        
        # Then run original validation for on-topic or potentially harmful queries
        return await original_validate_query(query)
    
    guard_instance.validate_query = enhanced_validate_query
    guard_instance.off_topic_handler = handler
    
    return guard_instance

# ============================================================================
# STANDALONE USAGE
# ============================================================================

def handle_off_topic(query: str) -> Tuple[bool, str]:
    """
    Standalone function to handle off-topic requests.
    Returns: (is_off_topic, response)
    """
    handler = OffTopicHandler()
    response = handler.generate_response(query)
    
    if response:
        return True, response
    return False, ""

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    test_queries = [
        "Can you give me a bedtime story for my son?",
        "Write me a poem about love",
        "How do I debug this Python code?",
        "What's your favorite color?",
        "Tell me about Journvax side effects",  # On-topic
        "Can you help me with my relationship?",
        "I need help creating a bedtime routine for my child",
    ]
    
    handler = OffTopicHandler()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = handler.generate_response(query)
        if response:
            print(f"Off-topic response: {response[:150]}...")
        else:
            print("On-topic - proceed with normal processing")