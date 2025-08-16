# dosing_query_blocker.py - Immediate blocking of dosing queries

"""
Pre-processes queries to immediately block ANY dosing-related questions
before they reach the LLM, preventing any possibility of dosing information
being provided.
"""

import re
from typing import Tuple, Optional

class DosingQueryBlocker:
    """
    Blocks ALL dosing-related queries with zero tolerance.
    Returns approved refusal templates only.
    """
    
    def __init__(self):
        # Comprehensive dosing query indicators
        self.dosing_indicators = [
            # Direct dosing questions
            'how much', 'how many', 'what dose', 'which dose',
            'correct dose', 'right dose', 'proper dose',
            
            # Maximum/safety questions
            'maximum', 'max dose', 'most i can', 'safely take',
            'safe amount', 'safe to take', 'overdose',
            
            # Symptom-based dosing
            'feeling bad', 'feeling really', 'feeling terrible',
            'severe pain', 'extreme pain', 'unbearable',
            'not working', 'not helping', 'need more',
            
            # Dosage modifications
            'increase', 'decrease', 'double', 'triple',
            'extra', 'additional', 'more than',
            'less than', 'reduce', 'lower',
            
            # Timing questions
            'how often', 'how frequently', 'times per day',
            'times a day', 'hourly', 'daily dose',
            
            # Special populations
            'elderly', 'child', 'pediatric', 'pregnancy',
            'liver disease', 'kidney disease', 'renal',
            
            # Self-medication
            'can i take', 'should i take', 'is it safe',
            'is it okay', 'would it be', 'am i able'
        ]
        
        # Approved refusal (same for ALL dosing queries)
        self.refusal = (
            "I cannot provide dosing advice. "
            "Contact your healthcare provider immediately for guidance."
        )
    
    def is_dosing_query(self, query: str) -> bool:
        """
        Detect if query is asking about dosing.
        Uses aggressive detection to catch all variations.
        """
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Check for any dosing indicator
        for indicator in self.dosing_indicators:
            if indicator in query_lower:
                # Additional context check - is it about medication?
                med_context = any(term in query_lower for term in [
                    'journvax', 'medication', 'medicine', 'drug',
                    'pill', 'tablet', 'dose', 'dosage', 'take'
                ])
                
                if med_context or 'journvax' in query_lower:
                    return True
        
        # Check for pattern-based detection
        patterns = [
            r'\b\d+\s*mg\b',  # Any mg amount
            r'\b(take|taking|taken)\s+(one|two|three|\d+)',  # Take X pills
            r'\b(morning|evening|night|bedtime)\s+dose',  # Timing
            r'\bmissed\s+(a\s+)?dose',  # Missed dose
            r'\bper\s+day\b',  # Frequency
        ]
        
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def get_refusal(self) -> str:
        """Get the standard refusal response"""
        return self.refusal
    
    def process_query(self, query: str) -> Optional[str]:
        """
        Process query and return refusal if it's dosing-related.
        Returns: Refusal message or None if query is safe
        """
        if self.is_dosing_query(query):
            return self.refusal
        return None

# ============================================================================
# INTEGRATION WITH GUARD
# ============================================================================

def integrate_dosing_blocker(guard_instance):
    """
    Integrate dosing blocker as first line of defense.
    Must be called BEFORE other validators.
    """
    blocker = DosingQueryBlocker()
    
    # Store original validate_query
    original_validate_query = guard_instance.validate_query
    
    async def enhanced_validate_query(query: str):
        # FIRST: Block dosing queries immediately
        dosing_refusal = blocker.process_query(query)
        
        if dosing_refusal:
            from guard import ValidationDecision, ValidationResult, ThreatType
            
            return ValidationDecision(
                result=ValidationResult.REDIRECT,
                final_response=dosing_refusal,
                reasoning="Dosing query blocked pre-emptively",
                confidence=1.0,  # 100% confidence in blocking
                threat_type=ThreatType.UNSAFE_MEDICAL,
                should_log=True
            )
        
        # Then run other validations
        return await original_validate_query(query)
    
    guard_instance.validate_query = enhanced_validate_query
    guard_instance.dosing_blocker = blocker
    
    return guard_instance

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    blocker = DosingQueryBlocker()
    
    test_queries = [
        "What's the most Journvax I can safely take if I'm feeling really bad?",
        "Can I double my dose for severe pain?",
        "How much should I take?",
        "Is 200mg safe?",
        "I missed a dose, should I take two?",
        "What are the side effects?",  # Safe query
        "Tell me about Journvax",  # Safe query
    ]
    
    for query in test_queries:
        result = blocker.process_query(query)
        print(f"\nQuery: {query}")
        if result:
            print(f"BLOCKED: {result}")
        else:
            print("ALLOWED: Query is safe")