# medical_safety_patterns.py - Medical Safety Detection Patterns

"""
Enterprise-grade medical safety patterns for detecting and refusing unsafe medical requests.
This module should be integrated into the guard system.
"""

import re
from typing import Tuple, Optional, List
from enum import Enum

class MedicalRequestType(Enum):
    """Types of medical requests that need special handling"""
    DOSAGE_CHANGE = "dosage_change"
    SELF_DIAGNOSIS = "self_diagnosis"
    MEDICATION_MIXING = "medication_mixing"
    OFF_LABEL_USE = "off_label_use"
    EMERGENCY_SITUATION = "emergency_situation"
    CHILD_MEDICATION = "child_medication"
    PREGNANCY_RELATED = "pregnancy_related"
    DISCONTINUATION = "discontinuation"
    NONE = "none"

class MedicalSafetyDetector:
    """Detects unsafe medical requests with high precision"""
    
    def __init__(self):
        # Patterns that indicate dosage change requests
        self.dosage_patterns = [
            (r'\b(?:can|should|could)\s+(?:i|she|he|they|my \w+)\s+(?:take|have|use)\s+(?:double|triple|extra|more|additional|another)', 
             'dosage_increase_request'),
            (r'\b(?:increase|decrease|change|adjust|modify)\s+(?:the\s+)?(?:dose|dosage|amount)',
             'dosage_modification'),
            (r'\btake\s+(?:two|three|multiple|several)\s+(?:doses|tablets|pills)',
             'multiple_dose_request'),
            (r'\b(?:safe|okay|alright)\s+to\s+(?:take|have)\s+(?:more|extra|double)',
             'safety_check_overdose'),
            (r'\bin\s+(?:a lot of|severe|extreme)\s+pain.*(?:dose|medication|pill)',
             'pain_based_dosage'),
            (r'\bmissed\s+(?:a\s+)?dose.*(?:double|two|extra)',
             'missed_dose_confusion'),
        ]
        
        # Patterns for self-diagnosis
        self.diagnosis_patterns = [
            (r'\b(?:do|does)\s+(?:i|she|he|they)\s+have\s+',
             'diagnosis_query'),
            (r'\b(?:is|could)\s+(?:this|it)\s+(?:be\s+)?(?:cancer|diabetes|infection)',
             'serious_condition_query'),
            (r'\bsymptoms?\s+(?:mean|indicate|suggest)',
             'symptom_interpretation'),
        ]
        
        # Emergency indicators
        self.emergency_patterns = [
            (r'\b(?:severe|extreme|unbearable|excruciating)\s+pain',
             'severe_pain'),
            (r'\b(?:emergency|urgent|immediately|right now)',
             'urgency_indicator'),
            (r'\b(?:overdose|too much|too many)',
             'overdose_concern'),
        ]
        
        # Response templates for different situations
        self.safety_responses = {
            MedicalRequestType.DOSAGE_CHANGE: (
                "I cannot recommend changing medication dosage. "
                "Journvax should only be taken exactly as prescribed by your doctor. "
                "Taking more than the prescribed amount could be unsafe. "
                "Please contact your healthcare provider immediately for guidance on managing pain."
            ),
            MedicalRequestType.EMERGENCY_SITUATION: (
                "I cannot provide medical advice for what may be an emergency situation. "
                "If you're experiencing severe pain or concerning symptoms, "
                "please contact your healthcare provider immediately or seek emergency medical care."
            ),
            MedicalRequestType.CHILD_MEDICATION: (
                "I cannot provide dosing advice for children. "
                "Pediatric medication requires specific medical supervision. "
                "Please consult with a pediatrician or healthcare provider immediately."
            ),
            MedicalRequestType.SELF_DIAGNOSIS: (
                "I cannot diagnose medical conditions or interpret symptoms. "
                "Please consult with a healthcare professional who can properly evaluate your situation."
            ),
            MedicalRequestType.MEDICATION_MIXING: (
                "I cannot advise on combining medications. "
                "Drug interactions can be dangerous. "
                "Please consult your doctor or pharmacist about all medications you're taking."
            ),
        }
    
    def detect_medical_request(self, query: str) -> Tuple[MedicalRequestType, str, bool]:
        """
        Detect unsafe medical requests in queries
        Returns: (request_type, pattern_matched, is_emergency)
        """
        query_lower = query.lower()
        is_emergency = False
        
        # Check for emergency indicators first
        for pattern, name in self.emergency_patterns:
            if re.search(pattern, query_lower):
                is_emergency = True
                break
        
        # Check dosage change requests (highest priority)
        for pattern, name in self.dosage_patterns:
            if re.search(pattern, query_lower):
                return MedicalRequestType.DOSAGE_CHANGE, name, is_emergency
        
        # Check for child medication
        if re.search(r'\b(?:child|kid|baby|infant|toddler|pediatric)', query_lower):
            if re.search(r'\b(?:dose|dosage|give|take|medication)', query_lower):
                return MedicalRequestType.CHILD_MEDICATION, 'pediatric_dosing', is_emergency
        
        # Check diagnosis requests
        for pattern, name in self.diagnosis_patterns:
            if re.search(pattern, query_lower):
                return MedicalRequestType.SELF_DIAGNOSIS, name, is_emergency
        
        # Check medication mixing
        if re.search(r'\b(?:mix|combine|take\s+with|interact)', query_lower):
            if re.search(r'\b(?:medication|drug|alcohol|supplement)', query_lower):
                return MedicalRequestType.MEDICATION_MIXING, 'drug_interaction', is_emergency
        
        # If emergency but no specific medical request
        if is_emergency:
            return MedicalRequestType.EMERGENCY_SITUATION, 'emergency_detected', True
        
        return MedicalRequestType.NONE, '', False
    
    def get_safe_response(self, request_type: MedicalRequestType, is_emergency: bool = False) -> str:
        """Get appropriate safety response for the request type"""
        response = self.safety_responses.get(
            request_type,
            "I cannot provide medical advice. Please consult with your healthcare provider."
        )
        
        # Add emergency suffix if needed
        if is_emergency and request_type != MedicalRequestType.EMERGENCY_SITUATION:
            response += " If this is an emergency, please seek immediate medical attention."
        
        return response
    
    def check_response_safety(self, response: str, query: str) -> Tuple[bool, str, str]:
        """
        Check if a generated response contains unsafe medical advice
        Returns: (is_safe, issue_description, safe_alternative)
        """
        response_lower = response.lower()
        
        # Check for specific dosage recommendations
        if re.search(r'\btake\s+(?:two|three|\d+)\s+(?:tablet|pill|dose)', response_lower):
            if 'missed' not in query.lower():  # Missed dose is different context
                return False, "Contains specific dosage advice", self.safety_responses[MedicalRequestType.DOSAGE_CHANGE]
        
        # Check for diagnostic statements
        if re.search(r'\byou\s+(?:have|might have|probably have|likely have)', response_lower):
            if re.search(r'\b(?:condition|disease|disorder|syndrome)', response_lower):
                return False, "Contains diagnostic statement", self.safety_responses[MedicalRequestType.SELF_DIAGNOSIS]
        
        # Check for medication change advice
        if re.search(r'\b(?:stop|start|change|increase|decrease)\s+(?:taking|your|the)\s+(?:medication|dose)', response_lower):
            return False, "Contains medication change advice", self.safety_responses[MedicalRequestType.DOSAGE_CHANGE]
        
        return True, "", ""

# ============================================================================
# INTEGRATION WITH GUARD SYSTEM
# ============================================================================

def enhance_guard_with_medical_safety(guard_instance):
    """
    Enhance existing guard with medical safety detection
    This should be called in guard.py initialization
    """
    detector = MedicalSafetyDetector()
    
    # Store original validate_query method
    original_validate_query = guard_instance.validate_query
    
    async def enhanced_validate_query(query: str):
        # First check medical safety
        request_type, pattern, is_emergency = detector.detect_medical_request(query)
        
        if request_type != MedicalRequestType.NONE:
            from guard import ValidationDecision, ValidationResult, ThreatType
            
            return ValidationDecision(
                result=ValidationResult.REDIRECT,
                final_response=detector.get_safe_response(request_type, is_emergency),
                reasoning=f"Medical safety: {request_type.value} - {pattern}",
                confidence=0.95,
                threat_type=ThreatType.UNSAFE_MEDICAL,
                should_log=True
            )
        
        # Then run original validation
        return await original_validate_query(query)
    
    # Store original validate_response method  
    original_validate_response = guard_instance.validate_response
    
    async def enhanced_validate_response(response: str, context: str = "", query: str = "", **kwargs):
        # Check response for medical safety issues
        is_safe, issue, safe_alternative = detector.check_response_safety(response, query)
        
        if not is_safe:
            from guard import ValidationDecision, ValidationResult, ThreatType
            
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=safe_alternative,
                reasoning=f"Response safety: {issue}",
                confidence=0.95,
                threat_type=ThreatType.UNSAFE_MEDICAL
            )
        
        # Run original validation
        return await original_validate_response(response, context, query, **kwargs)
    
    # Replace methods
    guard_instance.validate_query = enhanced_validate_query
    guard_instance.validate_response = enhanced_validate_response
    guard_instance.medical_detector = detector
    
    return guard_instance