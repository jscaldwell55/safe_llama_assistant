# final_safety_hardening.py - Ultimate compliance enforcement

"""
Final layer of safety hardening to ensure responses match regulatory standards.
Prevents ANY form of medical advice, alternatives, or suggestions.
"""

import re
from typing import Tuple, List, Optional
from enum import Enum

# ============================================================================
# PROHIBITED PATTERNS
# ============================================================================

class ProhibitedContent(Enum):
    """Categories of prohibited content in responses"""
    OFFERING_ALTERNATIVES = "offering_alternatives"
    GENERAL_MEDICAL_ADVICE = "general_medical_advice"
    DOSAGE_IMPLICATIONS = "dosage_implications"
    VOLUNTEERING_HELP = "volunteering_help"
    SPECULATIVE_ADVICE = "speculative_advice"
    OTC_RECOMMENDATIONS = "otc_recommendations"

# ============================================================================
# STRICT RESPONSE VALIDATOR
# ============================================================================

class StrictResponseValidator:
    """
    Final validation layer ensuring absolutely no medical advice or suggestions.
    """
    
    def __init__(self):
        # Phrases that should NEVER appear in responses
        self.prohibited_phrases = {
            ProhibitedContent.OFFERING_ALTERNATIVES: [
                "would you like some suggestions",
                "can i help you explore",
                "alternative strategies",
                "other options",
                "might be worth considering",
                "let me suggest",
                "here are some options"
            ],
            ProhibitedContent.GENERAL_MEDICAL_ADVICE: [
                "it's often suggested",
                "general guidelines",
                "people can consume",
                "clear liquids",
                "stay hydrated",
                "get rest",
                "drink water",
                "based on general"
            ],
            ProhibitedContent.DOSAGE_IMPLICATIONS: [
                "adjust your dosage",
                "may need to adjust",
                "change your dose",
                "dosage adjustment",
                "modify your prescription"
            ],
            ProhibitedContent.VOLUNTEERING_HELP: [
                "would you like me to",
                "can i help you",
                "shall i look up",
                "let me find",
                "i can search for",
                "want me to"
            ],
            ProhibitedContent.SPECULATIVE_ADVICE: [
                "might help",
                "may help",
                "could try",
                "consider trying",
                "possibly beneficial",
                "potentially useful"
            ],
            ProhibitedContent.OTC_RECOMMENDATIONS: [
                "over-the-counter",
                "otc remedies",
                "supplement",
                "vitamin",
                "herbal",
                "natural remedy"
            ]
        }
        
        # Required elements for alcohol/interaction queries
        self.required_warnings = {
            "alcohol": "Alcohol can sometimes increase the risk of side effects when combined with medications",
            "otc": "Combining alcohol, medications, and OTC remedies can sometimes cause unexpected interactions",
            "emergency": "If you ever develop a sudden or severe headache, seek medical attention right away"
        }
    
    def validate_response(self, response: str, query: str) -> Tuple[bool, str, List[str]]:
        """
        Validate response for prohibited content.
        Returns: (is_compliant, corrected_response, violations)
        """
        response_lower = response.lower()
        query_lower = query.lower()
        violations = []
        
        # Check for prohibited phrases
        for category, phrases in self.prohibited_phrases.items():
            for phrase in phrases:
                if phrase in response_lower:
                    violations.append(f"{category.value}: '{phrase}'")
        
        # If violations found, return appropriate safe response
        if violations:
            corrected = self._get_safe_response(query_lower)
            return False, corrected, violations
        
        # Check if required warnings are missing for certain queries
        if "alcohol" in query_lower or "drink" in query_lower or "beer" in query_lower:
            if self.required_warnings["alcohol"] not in response:
                corrected = self._enhance_with_alcohol_warning(response)
                return False, corrected, ["missing_alcohol_warning"]
        
        return True, response, []
    
    def _get_safe_response(self, query: str) -> str:
        """
        Get appropriate safe response based on query type.
        """
        # Alcohol + medication query
        if any(term in query for term in ["alcohol", "drink", "beer", "wine", "hangover"]):
            if "otc" in query or "over-the-counter" in query:
                return (
                    "I can't recommend specific over-the-counter products while taking Journvax. "
                    "Combining alcohol, medications, and OTC remedies can sometimes cause unexpected interactions. "
                    "The safest step is to check with your doctor or pharmacist — they can advise you on "
                    "options that won't interfere with your prescription."
                )
            else:
                return (
                    "I can't advise you to take more medication than prescribed. "
                    "Alcohol can sometimes increase the risk of side effects when combined with medications, "
                    "so it's best not to change your dose on your own. "
                    "If you're worried about symptoms, the safest step is to talk to your doctor or pharmacist — "
                    "they can give guidance tailored to your situation."
                )
        
        # Default safe response
        return (
            "I cannot provide medical advice. "
            "Please consult your healthcare provider for guidance tailored to your situation."
        )
    
    def _enhance_with_alcohol_warning(self, response: str) -> str:
        """
        Add required alcohol warning to response.
        """
        warning = (
            " Alcohol can sometimes increase the risk of side effects when combined with medications. "
            "Please consult your healthcare provider for guidance."
        )
        
        # Remove any existing inadequate warnings
        response = re.sub(r'stay hydrated.*?\.', '', response, flags=re.IGNORECASE)
        response = re.sub(r'get rest.*?\.', '', response, flags=re.IGNORECASE)
        
        return response.rstrip('.') + '.' + warning

# ============================================================================
# RESPONSE TEMPLATES
# ============================================================================

class SafeResponseTemplates:
    """
    Pre-approved response templates for common scenarios.
    """
    
    ALCOHOL_INTERACTION = (
        "I can't advise you to change your medication dose. "
        "Alcohol can sometimes increase the risk of side effects when combined with medications. "
        "Please consult your healthcare provider for guidance tailored to your situation. "
        "If you develop sudden or severe symptoms, seek medical attention immediately."
    )
    
    OTC_COMBINATION = (
        "I can't recommend specific over-the-counter products while taking Journvax. "
        "Combining medications and OTC remedies can sometimes cause unexpected interactions. "
        "Please check with your doctor or pharmacist for options that won't interfere with your prescription."
    )
    
    GENERAL_REFUSAL = (
        "I cannot provide medical advice. "
        "Please consult your healthcare provider for guidance."
    )
    
    @classmethod
    def get_template(cls, query: str) -> str:
        """
        Get appropriate template based on query content.
        """
        query_lower = query.lower()
        
        if "otc" in query_lower or "over-the-counter" in query_lower:
            return cls.OTC_COMBINATION
        
        if any(term in query_lower for term in ["alcohol", "drink", "beer", "wine", "hangover"]):
            return cls.ALCOHOL_INTERACTION
        
        return cls.GENERAL_REFUSAL

# ============================================================================
# RESPONSE SANITIZER
# ============================================================================

class ResponseSanitizer:
    """
    Removes all forms of advice, suggestions, and speculation from responses.
    """
    
    def sanitize(self, response: str) -> str:
        """
        Remove all prohibited content from response.
        """
        # Remove offering help or alternatives
        patterns_to_remove = [
            r"Would you like.*?\?",
            r"Can I help you.*?\?",
            r"Let me.*?\.",
            r"I can.*?for you\.",
            r"alternative strategies.*?\.",
            r"other options.*?\.",
            r"considering some.*?\.",
        ]
        
        for pattern in patterns_to_remove:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)
        
        # Remove general medical advice
        advice_patterns = [
            r"it's often suggested that.*?\.",
            r"general guidelines.*?\.",
            r"stay hydrated.*?\.",
            r"get rest.*?\.",
            r"drink water.*?\.",
            r"clear liquids.*?\.",
        ]
        
        for pattern in advice_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and periods
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\.+', '.', response)
        response = response.strip()
        
        return response

# ============================================================================
# INTEGRATION
# ============================================================================

def enforce_final_safety(response: str, query: str) -> str:
    """
    Main enforcement function for final safety layer.
    """
    validator = StrictResponseValidator()
    sanitizer = ResponseSanitizer()
    
    # First sanitize obvious problems
    response = sanitizer.sanitize(response)
    
    # Then validate and correct if needed
    is_compliant, corrected, violations = validator.validate_response(response, query)
    
    if not is_compliant:
        # Log violations for monitoring
        if violations:
            print(f"Safety violations detected: {violations}")
        return corrected
    
    return response

# ============================================================================
# QUERY CLASSIFIER
# ============================================================================

class UnsafeQueryClassifier:
    """
    Classifies queries that should get template responses only.
    """
    
    def __init__(self):
        self.unsafe_patterns = [
            # Alcohol combinations
            r'\b(alcohol|drink|beer|wine|hangover).*?(journvax|medication)',
            r'\b(journvax|medication).*?(alcohol|drink|beer|wine)',
            
            # OTC combinations
            r'\b(otc|over.?the.?counter).*?(journvax|medication)',
            r'\b(journvax|medication).*?(otc|over.?the.?counter)',
            
            # Interaction queries
            r'\bcombine.*?with',
            r'\bmix.*?with',
            r'\binteract.*?with',
            
            # Self-medication
            r'\bmore.*?than.*?prescribed',
            r'\bextra.*?dose',
            r'\bincrease.*?dose',
        ]
    
    def is_unsafe(self, query: str) -> bool:
        """
        Check if query should get template response only.
        """
        query_lower = query.lower()
        
        for pattern in self.unsafe_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def get_template_response(self, query: str) -> Optional[str]:
        """
        Get template response if query is unsafe.
        """
        if self.is_unsafe(query):
            return SafeResponseTemplates.get_template(query)
        return None

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    test_cases = [
        {
            "query": "Can I take extra Journvax if I'm drinking beer?",
            "bad_response": "I can't advise you to take more medication. Would you like some suggestions for managing hangovers?",
            "expected": "Contains offering alternatives"
        },
        {
            "query": "What OTC can I take for hangovers with Journvax?",
            "bad_response": "Based on general guidelines, people can consume clear liquids like water.",
            "expected": "Contains general medical advice"
        }
    ]
    
    validator = StrictResponseValidator()
    
    for test in test_cases:
        is_compliant, corrected, violations = validator.validate_response(
            test["bad_response"], 
            test["query"]
        )
        
        print(f"\nQuery: {test['query']}")
        print(f"Original: {test['bad_response']}")
        print(f"Violations: {violations}")
        print(f"Corrected: {corrected}")