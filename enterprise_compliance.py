# enterprise_compliance.py - Strict compliance with enterprise safety standards

"""
Ensures all responses meet enterprise pharmaceutical compliance requirements.
Enforces deterministic safety policies and approved response templates.
"""

import re
from typing import Tuple, Optional, Dict, List
from enum import Enum
from dataclasses import dataclass

# ============================================================================
# COMPLIANCE REQUIREMENTS
# ============================================================================

class ComplianceViolation(Enum):
    """Types of compliance violations"""
    UNSAFE_MEDICAL = "unsafe_medical"
    UNGROUNDED_CLAIM = "ungrounded_claim"
    HEDGING_IN_REFUSAL = "hedging_in_refusal"
    SCOPE_CREEP = "scope_creep"
    MISSING_DISCLAIMER = "missing_disclaimer"
    NONE = "none"

# ============================================================================
# PRE-APPROVED TEMPLATES (No hedging, deterministic)
# ============================================================================

APPROVED_REFUSAL_TEMPLATES = {
    "dosing_advice": (
        "I cannot provide dosing advice. "
        "Take Journvax exactly as prescribed by your healthcare provider."
    ),
    
    "prescription_sharing": (
        "I cannot recommend sharing prescription medications. "
        "Each person must have their own prescription from a healthcare provider."
    ),
    
    "pediatric_without_rx": (
        "I cannot provide pediatric dosing information without a prescription. "
        "Consult a pediatrician for appropriate medical care."
    ),
    
    "illegal_violent": (
        "I cannot and will not provide information about illegal or harmful activities. "
        "Please seek appropriate resources for your needs."
    ),
    
    "sexual_content": (
        "I cannot create or discuss inappropriate content. "
        "I'm here to provide pharmaceutical information about Journvax."
    ),
    
    "medical_advice": (
        "I cannot provide medical advice. "
        "Consult your healthcare provider for medical decisions."
    ),
    
    "off_topic": (
        "I provide information specifically about Journvax. "
        "I cannot assist with that topic."
    )
}

# Approved disclaimer for medical content
MEDICAL_DISCLAIMER = (
    "This is not a complete list. See the Medication Guide for full information. "
    "If symptoms are severe, seek immediate medical care."
)

# ============================================================================
# COMPLIANCE VALIDATOR
# ============================================================================

class ComplianceValidator:
    """
    Validates responses against enterprise compliance standards.
    Ensures deterministic safety policies and proper disclaimers.
    """
    
    def __init__(self):
        # Hedging phrases that should NEVER appear in safety refusals
        self.hedging_phrases = [
            "i think", "maybe", "perhaps", "might be best",
            "generally recommended", "typically", "usually",
            "it seems", "appears to", "probably should",
            "you might want to", "you could consider"
        ]
        
        # Ungrounded comparison indicators
        self.comparison_phrases = [
            "compared to", "versus", "better than", "worse than",
            "more effective", "less effective", "%", "percent",
            "studies show", "research indicates", "data suggests"
        ]
        
        # Scope creep indicators (tangential info)
        self.tangential_topics = [
            "fertility", "contraceptive", "birth control", "28 days",
            "hormonal", "pregnancy category", "lactation"
        ]
        
        # Required disclaimer triggers
        self.medical_info_keywords = [
            "side effect", "adverse", "reaction", "symptom",
            "dosage", "administration", "interaction"
        ]
    
    def validate_refusal(self, response: str, refusal_type: str) -> Tuple[bool, str]:
        """
        Validate that a refusal response is compliant.
        Returns: (is_compliant, corrected_response)
        """
        response_lower = response.lower()
        
        # Check for hedging in refusal
        for hedge in self.hedging_phrases:
            if hedge in response_lower:
                # Replace with approved template
                approved = APPROVED_REFUSAL_TEMPLATES.get(
                    refusal_type, 
                    APPROVED_REFUSAL_TEMPLATES["medical_advice"]
                )
                return False, approved
        
        # Ensure refusal is clear and starts appropriately
        if not response_lower.startswith(("i cannot", "i will not", "i do not")):
            approved = APPROVED_REFUSAL_TEMPLATES.get(
                refusal_type,
                APPROVED_REFUSAL_TEMPLATES["medical_advice"]
            )
            return False, approved
        
        return True, response
    
    def validate_medical_content(self, response: str, query: str) -> Tuple[bool, str]:
        """
        Validate medical content meets compliance standards.
        Returns: (is_compliant, corrected_response)
        """
        response_lower = response.lower()
        query_lower = query.lower()
        violations = []
        
        # Check for ungrounded comparisons/numbers
        for phrase in self.comparison_phrases:
            if phrase in response_lower:
                violations.append(ComplianceViolation.UNGROUNDED_CLAIM)
        
        # Check for numbers without citation
        if re.search(r'\b\d+\s*(%|percent|mg|ml|hours|days)\b', response_lower):
            if "according to" not in response_lower and "per the" not in response_lower:
                violations.append(ComplianceViolation.UNGROUNDED_CLAIM)
        
        # Check for scope creep (tangential topics not in query)
        for topic in self.tangential_topics:
            if topic in response_lower and topic not in query_lower:
                violations.append(ComplianceViolation.SCOPE_CREEP)
        
        # Check for missing disclaimer on medical info
        has_medical_info = any(keyword in response_lower for keyword in self.medical_info_keywords)
        has_disclaimer = any(phrase in response_lower for phrase in [
            "not a complete list", "see the medication guide",
            "consult your healthcare provider", "seek medical"
        ])
        
        if has_medical_info and not has_disclaimer:
            violations.append(ComplianceViolation.MISSING_DISCLAIMER)
        
        # If violations found, return corrected response
        if violations:
            if ComplianceViolation.SCOPE_CREEP in violations:
                # Remove tangential information
                response = self._remove_tangential_info(response, query)
            
            if ComplianceViolation.MISSING_DISCLAIMER in violations:
                # Add required disclaimer
                response = response.rstrip('.') + ". " + MEDICAL_DISCLAIMER
            
            if ComplianceViolation.UNGROUNDED_CLAIM in violations:
                # Remove ungrounded claims
                response = self._remove_ungrounded_claims(response)
            
            return False, response
        
        return True, response
    
    def _remove_tangential_info(self, response: str, query: str) -> str:
        """Remove tangential information not directly asked about"""
        sentences = response.split('. ')
        query_lower = query.lower()
        filtered = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Keep if it doesn't contain tangential topics OR if topic was in query
            is_tangential = False
            for topic in self.tangential_topics:
                if topic in sentence_lower and topic not in query_lower:
                    is_tangential = True
                    break
            
            if not is_tangential:
                filtered.append(sentence)
        
        result = '. '.join(filtered)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def _remove_ungrounded_claims(self, response: str) -> str:
        """Remove or caveat ungrounded numerical claims"""
        # Remove sentences with uncited numbers/comparisons
        sentences = response.split('. ')
        filtered = []
        
        for sentence in sentences:
            has_number = bool(re.search(r'\b\d+\s*(%|percent|mg|ml|hours|days)\b', sentence))
            has_comparison = any(phrase in sentence.lower() for phrase in self.comparison_phrases)
            has_citation = "according to" in sentence.lower() or "per the" in sentence.lower()
            
            if (has_number or has_comparison) and not has_citation:
                # Skip this sentence or add caveat
                continue
            
            filtered.append(sentence)
        
        result = '. '.join(filtered)
        if result and not result.endswith('.'):
            result += '.'
        
        return result if result else "I cannot provide specific numbers without proper documentation."
    
    def enforce_compliance(self, response: str, query: str, is_refusal: bool = False) -> str:
        """
        Main compliance enforcement function.
        Returns: Compliant response
        """
        # For refusals, ensure no hedging and use templates
        if is_refusal:
            # Detect refusal type from response content
            response_lower = response.lower()
            refusal_type = "medical_advice"  # default
            
            if "dosing" in response_lower or "dosage" in response_lower:
                refusal_type = "dosing_advice"
            elif "sharing" in response_lower or "prescription" in response_lower:
                refusal_type = "prescription_sharing"
            elif "child" in response_lower or "pediatric" in response_lower:
                refusal_type = "pediatric_without_rx"
            
            is_compliant, corrected = self.validate_refusal(response, refusal_type)
            return corrected if not is_compliant else response
        
        # For medical content, ensure grounding and disclaimers
        is_compliant, corrected = self.validate_medical_content(response, query)
        return corrected if not is_compliant else response

# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def get_approved_refusal(category: str) -> str:
    """Get approved refusal template for a category"""
    return APPROVED_REFUSAL_TEMPLATES.get(
        category,
        APPROVED_REFUSAL_TEMPLATES["medical_advice"]
    )

def add_medical_disclaimer(response: str) -> str:
    """Add required medical disclaimer to response"""
    if MEDICAL_DISCLAIMER not in response:
        return response.rstrip('.') + ". " + MEDICAL_DISCLAIMER
    return response

def remove_hedging(text: str) -> str:
    """Remove hedging language from text"""
    validator = ComplianceValidator()
    for hedge in validator.hedging_phrases:
        text = text.replace(hedge, "")
    
    # Clean up double spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================================================
# RESPONSE BUILDER
# ============================================================================

class CompliantResponseBuilder:
    """Builds responses that meet all compliance requirements"""
    
    @staticmethod
    def build_refusal(category: str, add_redirect: bool = True) -> str:
        """Build a compliant refusal response"""
        base = APPROVED_REFUSAL_TEMPLATES.get(
            category,
            APPROVED_REFUSAL_TEMPLATES["medical_advice"]
        )
        
        if add_redirect and "healthcare provider" not in base:
            base += " Please consult your healthcare provider."
        
        return base
    
    @staticmethod
    def build_medical_info(facts: List[str], add_disclaimer: bool = True) -> str:
        """Build compliant medical information response"""
        response = " ".join(facts)
        
        if add_disclaimer:
            response = response.rstrip('.') + ". " + MEDICAL_DISCLAIMER
        
        return response
    
    @staticmethod
    def build_off_topic() -> str:
        """Build compliant off-topic response"""
        return APPROVED_REFUSAL_TEMPLATES["off_topic"]

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    validator = ComplianceValidator()
    
    # Test cases
    test_cases = [
        {
            "response": "I think maybe you should probably consult your doctor about dosing.",
            "query": "Can I double my dose?",
            "is_refusal": True,
            "expected_issue": "Hedging in refusal"
        },
        {
            "response": "Side effects include nausea and headache.",
            "query": "What are the side effects?",
            "is_refusal": False,
            "expected_issue": "Missing disclaimer"
        },
        {
            "response": "Journvax is 75% more effective than placebo.",
            "query": "How effective is Journvax?",
            "is_refusal": False,
            "expected_issue": "Ungrounded claim"
        },
        {
            "response": "Common side effects include nausea. Also, Journvax affects fertility for 28 days.",
            "query": "What are common side effects?",
            "is_refusal": False,
            "expected_issue": "Scope creep"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['expected_issue']} ---")
        print(f"Original: {test['response']}")
        
        compliant = validator.enforce_compliance(
            test['response'],
            test['query'],
            test['is_refusal']
        )
        
        print(f"Compliant: {compliant}")
        print(f"Fixed: {compliant != test['response']}")