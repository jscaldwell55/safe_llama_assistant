# medical_safety_patterns.py - Simplified Semantic Version

"""
Simplified medical safety detection using semantic categories rather than exhaustive patterns.
Focuses on core unsafe behaviors rather than specific phrases.
"""

import re
import logging
from typing import Tuple, Optional, List, Dict
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

class MedicalRequestType(Enum):
    """Core categories of unsafe medical requests"""
    DOSAGE_MODIFICATION = "dosage_modification"  # Any change to prescribed dosing
    PRESCRIPTION_MISUSE = "prescription_misuse"  # Sharing, no Rx, etc.
    MEDICAL_ADVICE = "medical_advice"  # Diagnosis, treatment decisions
    EMERGENCY = "emergency"  # Severe symptoms needing immediate care
    NONE = "none"

@dataclass
class MedicalSafetyResult:
    """Result of medical safety check"""
    request_type: MedicalRequestType
    confidence: float
    reasoning: str
    is_emergency: bool = False

# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class SimplifiedMedicalDetector:
    """
    Semantic detection of unsafe medical requests.
    Uses concept-based detection rather than exhaustive pattern matching.
    """
    
    def __init__(self):
        # Core concept indicators (not exhaustive lists)
        self.dosage_concepts = ['double', 'extra', 'more', 'increase', 'decrease', 'change', 'adjust', 'modify']
        self.sharing_concepts = ['share', 'sharing', 'borrow', 'lend', 'split', 'give', 'take someone else']
        self.prescription_concepts = ['without prescription', 'no prescription', 'not prescribed', "doesn't have prescription"]
        self.emergency_concepts = ['severe pain', 'emergency', 'urgent', 'immediately', 'right now', 'unbearable', 'excruciating']
        self.child_concepts = ['child', 'kid', 'baby', 'infant', 'toddler', 'pediatric', 'my son', 'my daughter']
        
        # Single, clear response for each category
        self.responses = {
            MedicalRequestType.DOSAGE_MODIFICATION: (
                "I cannot recommend changing medication dosage. "
                "Please take Journvax exactly as prescribed by your doctor. "
                "Contact your healthcare provider for any dosage questions."
            ),
            MedicalRequestType.PRESCRIPTION_MISUSE: (
                "I cannot recommend using prescription medication without a valid prescription "
                "or sharing medications between people. Each person needs their own prescription "
                "from a healthcare provider. Please consult a doctor for proper medical evaluation."
            ),
            MedicalRequestType.MEDICAL_ADVICE: (
                "I cannot provide medical advice or diagnosis. "
                "Please consult with your healthcare provider for medical decisions."
            ),
            MedicalRequestType.EMERGENCY: (
                "If you're experiencing severe symptoms, please contact your healthcare provider "
                "immediately or seek emergency medical care."
            ),
        }
        
        logger.info("SimplifiedMedicalDetector initialized")
    
    def detect_unsafe_request(self, query: str) -> MedicalSafetyResult:
        """
        Detect unsafe medical requests using semantic understanding.
        Returns: MedicalSafetyResult with details
        """
        if not query:
            return MedicalSafetyResult(MedicalRequestType.NONE, 0.0, "Empty query", False)
        
        query_lower = query.lower().strip()
        
        # Check for emergency first (highest priority)
        is_emergency = self._contains_concepts(query_lower, self.emergency_concepts)
        if is_emergency:
            return MedicalSafetyResult(
                MedicalRequestType.EMERGENCY, 
                0.95, 
                "Emergency indicators detected",
                True
            )
        
        # Check for child medication (high priority)
        if self._involves_child_medication(query_lower):
            return MedicalSafetyResult(
                MedicalRequestType.MEDICAL_ADVICE,
                0.95,
                "Pediatric medication requires medical supervision",
                is_emergency
            )
        
        # Check for dosage modification intent
        if self._indicates_dosage_change(query_lower):
            return MedicalSafetyResult(
                MedicalRequestType.DOSAGE_MODIFICATION,
                0.90,
                "Dosage modification request detected",
                is_emergency
            )
        
        # Check for prescription misuse
        if self._indicates_prescription_misuse(query_lower):
            return MedicalSafetyResult(
                MedicalRequestType.PRESCRIPTION_MISUSE,
                0.90,
                "Prescription misuse indicators detected",
                is_emergency
            )
        
        # Check for medical advice seeking
        if self._seeks_medical_advice(query_lower):
            return MedicalSafetyResult(
                MedicalRequestType.MEDICAL_ADVICE,
                0.85,
                "Medical advice request detected",
                is_emergency
            )
        
        return MedicalSafetyResult(MedicalRequestType.NONE, 0.0, "No unsafe patterns detected", False)
    
    def _contains_concepts(self, text: str, concepts: List[str]) -> bool:
        """Check if text contains any of the concept indicators"""
        return any(concept in text for concept in concepts)
    
    def _involves_child_medication(self, text: str) -> bool:
        """Check if query involves medication for children"""
        has_child = any(term in text for term in self.child_concepts)
        has_medication = any(term in text for term in ['dose', 'dosage', 'medication', 'journvax', 'give', 'take'])
        return has_child and has_medication
    
    def _indicates_dosage_change(self, text: str) -> bool:
        """Detect intent to modify dosage or ask about safe amounts"""
        # CRITICAL: Catch "safe amount" and "maximum" queries
        unsafe_dose_phrases = [
            'most i can', 'maximum', 'max dose', 'safely take',
            'how much can', 'how many can', 'safe to take',
            'feeling bad', 'feeling really', 'severe pain',
            'unbearable', 'excruciating'
        ]
        
        # Check for these phrases first
        if any(phrase in text for phrase in unsafe_dose_phrases):
            return True
        
        # Look for dosage words near change words
        has_dosage_term = any(term in text for term in ['dose', 'dosage', 'pill', 'tablet', 'medication', 'journvax'])
        has_change_term = any(term in text for term in self.dosage_concepts)
        
        # Also check for pain-based reasoning
        has_pain_reason = 'pain' in text and any(term in text for term in ['more', 'extra', 'double', 'additional'])
        
        # Check for missed dose confusion (people asking to double up)
        missed_dose_confusion = 'missed' in text and any(term in text for term in ['double', 'two', 'extra'])
        
        return (has_dosage_term and has_change_term) or has_pain_reason or missed_dose_confusion
    
    def _indicates_prescription_misuse(self, text: str) -> bool:
        """Detect prescription sharing or use without Rx"""
        # Sharing indicators
        if any(term in text for term in self.sharing_concepts):
            return True
        
        # No prescription indicators
        if 'prescription' in text or 'rx' in text.lower():
            negatives = ['without', 'no', "doesn't have", "hasn't got", 'until', 'before']
            return any(neg in text for neg in negatives)
        
        # Other misuse patterns
        misuse_phrases = ['not prescribed', 'someone else', 'my friend', 'leftover', 'extra pills']
        return any(phrase in text for phrase in misuse_phrases)
    
    def _seeks_medical_advice(self, text: str) -> bool:
        """Detect requests for medical advice vs information"""
        # Diagnosis seeking
        if re.search(r'\b(do|does|could|might) (i|he|she|they|my \w+) have\b', text):
            return True
        
        # Treatment decisions
        if re.search(r'\b(should|can|is it safe|is it okay|safe to|okay to) (i|we|he|she)', text):
            medical_terms = ['take', 'stop', 'start', 'mix', 'combine', 'use']
            return any(term in text for term in medical_terms)
        
        # Symptom interpretation
        if re.search(r'(symptom|side effect|reaction).*(mean|indicate|serious|dangerous|normal)', text):
            return True
        
        # Self-treatment
        return bool(re.search(r'(what|how) (can|should) i (do|take) for', text))
    
    def validate_response(self, response: str, query: str) -> Tuple[bool, str, str]:
        """
        Validate that response doesn't contain unsafe advice.
        Returns: (is_safe, issue_description, corrected_response_if_unsafe)
        """
        if not response:
            return True, "", response
        
        response_lower = response.lower()
        query_result = self.detect_unsafe_request(query)
        
        # If query was unsafe, ensure response doesn't provide the requested info
        if query_result.request_type != MedicalRequestType.NONE:
            # Check for problematic approval patterns
            approval_patterns = [
                'you can', 'you could', 'you might', 'it should be',
                'generally recommended', 'typically', 'usually',
                'it\'s fine', 'it\'s okay', 'no problem'
            ]
            
            # Check if response seems to be giving advice rather than refusing
            for pattern in approval_patterns:
                if pattern in response_lower:
                    logger.warning(f"Response contains approval pattern '{pattern}' for unsafe request")
                    return False, f"Implicit approval detected: {pattern}", self.responses[query_result.request_type]
        
        # Check for specific unsafe content regardless of query
        unsafe_patterns = [
            (r'\btake \d+ (tablet|pill|dose)', MedicalRequestType.DOSAGE_MODIFICATION, "Specific dosage instruction"),
            (r'\b(stop|start) taking\b', MedicalRequestType.MEDICAL_ADVICE, "Medication change advice"),
            (r'\byou (have|might have|probably have|likely have) \w+ (condition|disease|disorder)', 
             MedicalRequestType.MEDICAL_ADVICE, "Diagnostic statement"),
            (r'\b(increase|decrease|change) (your|the) (dose|dosage)', 
             MedicalRequestType.DOSAGE_MODIFICATION, "Dosage modification advice"),
        ]
        
        for pattern, request_type, description in unsafe_patterns:
            if re.search(pattern, response_lower):
                logger.warning(f"Response contains unsafe pattern: {description}")
                return False, description, self.responses[request_type]
        
        return True, "", response
    
    def get_safe_response(self, request_type: MedicalRequestType, is_emergency: bool = False) -> str:
        """Get appropriate safe response for request type"""
        base_response = self.responses.get(
            request_type, 
            "I cannot provide medical advice. Please consult with your healthcare provider."
        )
        
        # Add emergency suffix if needed
        if is_emergency and request_type != MedicalRequestType.EMERGENCY:
            base_response += " If this is an emergency, please seek immediate medical attention."
        
        return base_response

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_medical_safety_guard() -> SimplifiedMedicalDetector:
    """Factory function to create medical safety detector"""
    return SimplifiedMedicalDetector()

# ============================================================================
# STANDALONE VALIDATION FUNCTION
# ============================================================================

def check_medical_safety(query: str) -> Tuple[bool, str]:
    """
    Standalone function to check if a query is medically safe.
    Returns: (is_safe, response_if_unsafe)
    """
    detector = SimplifiedMedicalDetector()
    result = detector.detect_unsafe_request(query)
    
    if result.request_type != MedicalRequestType.NONE and result.confidence > 0.8:
        return False, detector.get_safe_response(result.request_type, result.is_emergency)
    
    return True, ""

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the detector with various queries
    test_queries = [
        "Can my wife and I share our journvax until she gets her own prescription?",
        "Can I take double dose if I'm in severe pain?",
        "What are the side effects of Journvax?",  # Safe query
        "My child needs medication, what dose should I give?",
        "I missed a dose, should I take two?",
        "Tell me about Journvax usage",  # Safe query
    ]
    
    detector = create_medical_safety_guard()
    
    for query in test_queries:
        result = detector.detect_unsafe_request(query)
        print(f"\nQuery: {query}")
        print(f"Result: {result.request_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Emergency: {result.is_emergency}")
        if result.request_type != MedicalRequestType.NONE:
            print(f"Response: {detector.get_safe_response(result.request_type, result.is_emergency)[:100]}...")