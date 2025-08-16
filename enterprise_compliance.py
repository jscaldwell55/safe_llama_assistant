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
# PRE-APPROVED TEMPLATES (No hedging, deterministic, NO medical details)
# ============================================================================

APPROVED_REFUSAL_TEMPLATES = {
    "dosing_advice": (
        "I cannot provide dosing advice. "
        "Contact your healthcare provider immediately for guidance."
    ),
    
    "maximum_dose": (
        "I cannot advise on maximum doses or safe amounts. "
        "Contact your healthcare provider immediately for guidance."
    ),
    
    "prescription_sharing": (
        "I cannot recommend sharing prescription medications. "
        "Each person must have their own prescription from a healthcare provider."
    ),
    
    "pediatric_without_rx": (
        "I cannot provide pediatric dosing information. "
        "Consult a pediatrician immediately."
    ),
    
    "illegal_violent": (
        "I cannot provide information about illegal or harmful activities."
    ),
    
    "sexual_content": (
        "I cannot create or discuss inappropriate content. "
        "I provide pharmaceutical information about Journvax."
    ),
    
    "medical_advice": (
        "I cannot provide medical advice. "
        "Contact your healthcare provider."
    ),
    
    "self_medication": (
        "I cannot advise on self-medication or changing your dose. "
        "Contact your healthcare provider immediately."
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
        
        # Risk minimization phrases that should NEVER appear
        self.risk_minimization = [
            "don't worry", "no need to worry", "it's fine",
            "should be okay", "nothing to worry about",
            "err on the side of caution", "better safe than sorry"
        ]
        
        # Unsanctioned guidance phrases
        self.unsanctioned_guidance = [
            "have a light meal", "consider having", "try eating",
            "stick to your schedule", "regular schedule",
            "unless otherwise advised", "before taking",
            "after taking", "with food", "without food",
            "empty stomach", "grapefruit", "juice"
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
    
    def validate_response(self, response: str, query: str) -> Tuple[bool, str, str]:
        """
        Check if a generated response contains unsafe medical advice or dosing info.
        Returns: (is_safe, issue_description, safe_alternative)
        """
        response_lower = response.lower()
        query_lower = query.lower()
        
        # CRITICAL: Check for ANY dosing information in response to unsafe queries
        unsafe_query_indicators = [
            'how much', 'maximum', 'safely take', 'double dose', 
            'extra', 'more than', 'increase', 'feeling bad', 'severe pain'
        ]
        
        is_unsafe_query = any(indicator in query_lower for indicator in unsafe_query_indicators)
        
        if is_unsafe_query:
            # Check if response contains ANY medical details
            medical_details = [
                '100mg', '100 mg', 'starting dose', 'recommended dose',
                'empty stomach', 'swallow whole', 'before eating', 'after eating',
                'tablet', 'chewing', 'crushing', 'side effect'
            ]
            
            contains_medical = any(detail in response_lower for detail in medical_details)
            
            if contains_medical:
                # Response is mixing refusal with medical content - NOT compliant
                return False, "Contains medical details in response to unsafe query", APPROVED_REFUSAL_TEMPLATES["maximum_dose"]
        
        # Check for "we should" language (prescriptive)
        if re.search(r'\bwe should\b', response_lower):
            return False, "Contains prescriptive language", APPROVED_REFUSAL_TEMPLATES["dosing_advice"]
        
        # Check for specific mg amounts
        if re.search(r'\b\d+\s*mg\b', response_lower):
            # Unless it's clearly labeled as "according to the label" or similar
            if not re.search(r'(according to|per the|as stated in) (the )?(label|guide|documentation)', response_lower):
                return False, "Contains specific dosing amounts", APPROVED_REFUSAL_TEMPLATES["dosing_advice"]
        
        # Check for administration instructions mixed with refusals
        admin_instructions = [
            'empty stomach', 'with food', 'without food',
            'swallow whole', 'do not crush', 'do not chew',
            'before eating', 'after eating', 'with water'
        ]
        
        if any(instruction in response_lower for instruction in admin_instructions):
            # Check if this is in response to a dosing/safety question
            if is_unsafe_query:
                return False, "Contains administration instructions for unsafe query", APPROVED_REFUSAL_TEMPLATES["dosing_advice"]
        
        return True, "", ""
    
    def validate_medical_content(self, response: str, query: str) -> Tuple[bool, str]:
        """
        Validate medical content meets compliance standards.
        Returns: (is_compliant, corrected_response)
        """
        response_lower = response.lower()
        query_lower = query.lower()
        violations = []
        
        # Check for risk minimization
        for phrase in self.risk_minimization:
            if phrase in response_lower:
                return False, "I cannot provide administration guidance. Contact your healthcare provider immediately."
        
        # Check for unsanctioned guidance
        for phrase in self.unsanctioned_guidance:
            if phrase in response_lower:
                return False, "I cannot provide administration guidance. Contact your healthcare provider immediately."
        
        # Check for hedging
        for hedge in self.hedging_phrases:
            if hedge in response_lower:
                return False, "I cannot provide medical advice. Contact your healthcare provider."
        
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
        response_lower = response.lower()
        query_lower = query.lower()
        
        # FINAL SAFETY: Immediate template replacement for problematic patterns
        problematic_offers = [
            "would you like some suggestions",
            "can i help you explore",
            "alternative strategies",
            "other options",
            "let me suggest",
            "shall i look up"
        ]
        
        for phrase in problematic_offers:
            if phrase in response_lower:
                # Return safe template based on query type
                if "alcohol" in query_lower or "drink" in query_lower:
                    return (
                        "I can't advise you to change your medication dose. "
                        "Alcohol can sometimes increase the risk of side effects when combined with medications. "
                        "Please consult your healthcare provider for guidance."
                    )
                elif "otc" in query_lower or "over-the-counter" in query_lower:
                    return (
                        "I can't recommend specific over-the-counter products while taking Journvax. "
                        "Combining medications and OTC remedies can sometimes cause unexpected interactions. "
                        "Please check with your doctor or pharmacist."
                    )
                else:
                    return "I cannot provide medical advice. Contact your healthcare provider."
        
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
            if not is_compliant:
                return corrected
        
        # For medical content, ensure grounding and disclaimers
        is_compliant, corrected = self.validate_medical_content(response, query)
        
        # FINAL SAFETY: Remove any remaining problematic content
        if is_compliant:
            # Sanitize general medical advice
            advice_patterns = [
                r"it's often suggested that.*?\.",
                r"general guidelines.*?\.",
                r"stay hydrated.*?\.",
                r"get rest.*?\.",
                r"drink water.*?\.",
                r"clear liquids.*?\.",
            ]
            
            for pattern in advice_patterns:
                corrected = re.sub(pattern, "", corrected, flags=re.IGNORECASE)
            
            # Clean up
            corrected = re.sub(r'\s+', ' ', corrected)
            corrected = corrected.strip()
            
            # If we removed too much, return safe default
            if len(corrected) < 20:
                return "I cannot provide medical advice. Contact your healthcare provider."
        
        return corrected

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