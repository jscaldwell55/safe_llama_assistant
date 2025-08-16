# guard.py - Enhanced Version with Sophisticated Safety Detection

import logging
import re
import json
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from config import (
    ENABLE_GUARD,
    SEMANTIC_SIMILARITY_THRESHOLD,
    USE_LLM_GUARD,
    DEFAULT_FALLBACK_MESSAGE
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED VALIDATION RESULTS
# ============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REDIRECT = "redirect"  # New: for redirecting dangerous queries

class ThreatType(Enum):
    NONE = "none"
    VIOLENCE = "violence"
    INAPPROPRIATE = "inappropriate"
    UNSAFE_MEDICAL = "unsafe_medical"
    MIXED_MALICIOUS = "mixed_malicious"
    OFF_TOPIC = "off_topic"

@dataclass
class ValidationDecision:
    result: ValidationResult
    final_response: str
    reasoning: str
    confidence: float = 0.0
    threat_type: ThreatType = ThreatType.NONE
    should_log: bool = False  # For security monitoring

# ============================================================================
# ENHANCED GUARD WITH EARLY DETECTION
# ============================================================================

class EnhancedGuard:
    """
    Sophisticated guard with:
    1. Early malicious query detection
    2. Proper handling of mixed queries
    3. Consistent response formatting
    4. Better transparency about limitations
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.similarity_threshold = SEMANTIC_SIMILARITY_THRESHOLD
        self.use_llm = USE_LLM_GUARD
        self.embedding_model = None
        self.medical_detector = None  # Will be initialized below
        self.off_topic_handler = None  # Will be initialized below
        self.compliance_validator = None  # Will be initialized below
        
        # Enhanced threat patterns
        self.threat_patterns = {
            ThreatType.VIOLENCE: [
                (r'\b(?:build|make|create|construct)(?:ing)?\s+(?:a\s+)?(?:bomb|explosive|weapon)', 'explosive_creation'),
                (r'\b(?:how to|ways to|methods to)\s+(?:harm|hurt|kill|poison)', 'harm_instructions'),
                (r'\b(?:suicide|self.?harm|end(?:ing)?\s+(?:my\s+)?life)', 'self_harm'),
                (r'\b(?:terror|attack|destroy|damage)\s+(?:plan|method|instruction)', 'destructive_intent'),
            ],
            ThreatType.INAPPROPRIATE: [
                (r'\b(?:sexy|erotic|sexual|nude|naked|porn)', 'sexual_content'),
                (r'\b(?:write|tell|create)\s+(?:me\s+)?(?:a\s+)?(?:sexy|erotic|adult)\s+(?:story|tale)', 'inappropriate_story'),
                (r'\b(?:romance|romantic|intimate)\s+(?:between|with|involving)', 'inappropriate_relationship'),
            ],
            ThreatType.UNSAFE_MEDICAL: [
                (r'\b(?:without|no)\s+(?:a\s+)?prescription', 'no_prescription'),
                (r'\b(?:my\s+)?(?:kid|child|baby|infant|toddler)\s+(?:take|try|use)', 'child_medication'),
                (r'\b(?:overdose|lethal\s+dose|maximum\s+dose)', 'dangerous_dosage'),
                (r'\b(?:mix|combine)\s+with\s+(?:alcohol|drugs|medication)', 'dangerous_combination'),
                (r'\bwhat\s+(?:dose|dosage|amount)\s+(?:should|to|for)', 'dosage_recommendation'),
            ]
        }
        
        # Critical safety patterns for response validation
        self.response_safety_patterns = [
            (r'\b(?:take|inject|consume)\s+\d+\s*(?:mg|ml|pills)\s+(?:every|daily|twice)', 'specific_dosage_instruction'),
            (r'\byou (?:definitely|certainly) have\b', 'diagnostic_claim'),
            (r'\b(?:stop|start) taking (?:journvax|your medication) immediately\b', 'medication_directive'),
            (r'\b(?:crush|snort|inject|smoke) (?:the|your|journvax)', 'dangerous_administration'),
        ]
        
        # Response templates for different threat types
        self.threat_responses = {
            ThreatType.VIOLENCE: "I cannot and will not provide information about creating weapons, explosives, or causing harm. If you're experiencing thoughts of self-harm, please contact a crisis helpline immediately: 988 (Suicide & Crisis Lifeline) or text HOME to 741741.",
            ThreatType.INAPPROPRIATE: "I cannot create inappropriate or sexual content. I'm here to provide helpful information about Journvax and pharmaceutical topics. How can I assist you with that instead?",
            ThreatType.UNSAFE_MEDICAL: "I cannot provide medical advice or dosage recommendations. This medication requires a prescription and should only be used under the guidance of a healthcare professional. Please consult with a doctor or pharmacist for personalized medical advice.",
            ThreatType.MIXED_MALICIOUS: "I've detected potentially harmful content in your query. I can only provide safe, appropriate information about Journvax. Please rephrase your question if you need legitimate pharmaceutical information.",
            ThreatType.OFF_TOPIC: "I'm specifically designed to provide information about Journvax. I don't have information about that topic. Is there something about Journvax I can help you with?"
        }
        
        self._load_embedding_model()
        self._initialize_medical_safety()
        self._initialize_off_topic_handler()
        self._initialize_compliance_validator()
    
    def _load_embedding_model(self):
        """Load embedding model for similarity check"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Embedding model loaded for enhanced guard")
        except Exception as e:
            logger.warning(f"Could not load embedding model for guard: {e}")
            self.embedding_model = None
    
    def _initialize_medical_safety(self):
        """Initialize medical safety detection"""
        try:
            from medical_safety_patterns import MedicalSafetyDetector
            self.medical_detector = MedicalSafetyDetector()
            logger.info("Medical safety detector initialized")
        except ImportError:
            # If medical_safety_patterns doesn't exist, create inline version
            self._create_inline_medical_detector()
    
    def _create_inline_medical_detector(self):
        """Create inline medical detector if module not found"""
        class InlineMedicalDetector:
            def detect_medical_request(self, query):
                query_lower = query.lower()
                
                # Critical patterns for dosage changes
                dosage_patterns = [
                    r'\b(?:can|should|could).*(?:take|have|use).*(?:double|triple|extra|more)',
                    r'\b(?:increase|decrease|change).*(?:dose|dosage)',
                    r'\bmissed.*dose.*(?:double|two|extra)',
                    r'\bin.*pain.*(?:more|extra|double).*(?:dose|medication)',
                ]
                
                for pattern in dosage_patterns:
                    if re.search(pattern, query_lower):
                        return ThreatType.UNSAFE_MEDICAL, pattern, False
                
                return ThreatType.NONE, '', False
            
            def get_safe_response(self, request_type, is_emergency=False):
                return (
                    "I cannot recommend changing medication dosage. "
                    "Journvax should only be taken exactly as prescribed by your doctor. "
                    "Please contact your healthcare provider for guidance."
                )
        
    def _initialize_off_topic_handler(self):
        """Initialize off-topic request handler"""
        try:
            from off_topic_handler import OffTopicHandler
            self.off_topic_handler = OffTopicHandler()
            logger.info("Off-topic handler initialized")
        except ImportError:
            # Create inline handler if module not found
            self._create_inline_off_topic_handler()
    
    def _create_inline_off_topic_handler(self):
        """Create inline off-topic handler if module not found"""
        class InlineOffTopicHandler:
            def generate_response(self, query):
                query_lower = query.lower()
                
                # Handle bedtime story requests
                if 'bedtime' in query_lower and 'story' in query_lower:
                    return (
                        "I'm not able to tell bedtime stories, but I can share some tips for "
                        "creating a calming bedtime routine. Many parents find that keeping "
                        "a consistent schedule — like reading a favorite book, dimming the lights, "
                        "or playing gentle music — helps children settle down more easily."
                    )
                
                # Handle other creative requests
                if any(word in query_lower for word in ['story', 'poem', 'song', 'write me']):
                    return (
                        "I'm specifically designed to provide information about Journvax, "
                        "so I'm not able to create creative content. Is there anything about "
                        "Journvax I can help you with?"
                    )
                
                return None
        
        self.off_topic_handler = InlineOffTopicHandler()
        logger.info("Inline off-topic handler created")
    
    def _initialize_compliance_validator(self):
        """Initialize compliance validator"""
        try:
            from enterprise_compliance import ComplianceValidator
            self.compliance_validator = ComplianceValidator()
            logger.info("Compliance validator initialized")
        except ImportError:
            # Create inline validator if module not found
            self._create_inline_compliance_validator()
    
    def _create_inline_compliance_validator(self):
        """Create inline compliance validator"""
        class InlineComplianceValidator:
            def enforce_compliance(self, response, query, is_refusal=False):
                # Basic compliance - remove hedging from refusals
                if is_refusal:
                    hedges = ["maybe", "perhaps", "might", "generally", "typically"]
                    for hedge in hedges:
                        response = response.replace(hedge, "")
                    response = response.replace("  ", " ").strip()
                
                # Add disclaimer for medical content
                if any(word in response.lower() for word in ["side effect", "symptom", "dosage"]):
                    if "not a complete list" not in response:
                        response += " This is not a complete list. See the Medication Guide for full information."
                
                return response
        
        self.compliance_validator = InlineComplianceValidator()
        logger.info("Inline compliance validator created")
    
    def _initialize_final_safety(self):
        """Initialize final safety hardening"""
        try:
            from final_safety_hardening import StrictResponseValidator, ResponseSanitizer, UnsafeQueryClassifier
            self.final_safety = {
                'validator': StrictResponseValidator(),
                'sanitizer': ResponseSanitizer(),
                'classifier': UnsafeQueryClassifier()
            }
            logger.info("Final safety hardening initialized")
        except ImportError:
            # Create inline version
            self._create_inline_final_safety()
    
    def _create_inline_final_safety(self):
        """Create inline final safety"""
        class InlineFinalSafety:
            def enforce(self, response, query):
                # Remove problematic phrases
                prohibited = [
                    "would you like some suggestions",
                    "can i help you explore",
                    "alternative strategies",
                    "stay hydrated",
                    "get rest"
                ]
                
                response_lower = response.lower()
                for phrase in prohibited:
                    if phrase in response_lower:
                        # Return safe template
                        if "alcohol" in query.lower():
                            return (
                                "I can't advise you to change your medication dose. "
                                "Alcohol can sometimes increase the risk of side effects when combined with medications. "
                                "Please consult your healthcare provider for guidance."
                            )
                        return "I cannot provide medical advice. Please consult your healthcare provider."
                
                return response
        
        self.final_safety = {'enforce': InlineFinalSafety().enforce}
        logger.info("Inline final safety created")
    
    def detect_query_threats(self, query: str) -> Tuple[ThreatType, str]:
        """
        Early detection of threats in user queries
        Returns: (threat_type, pattern_name)
        """
        query_lower = query.lower()
        
        # Check each threat category
        for threat_type, patterns in self.threat_patterns.items():
            for pattern, pattern_name in patterns:
                if re.search(pattern, query_lower):
                    logger.warning(f"Threat detected - Type: {threat_type.value}, Pattern: {pattern_name}")
                    return threat_type, pattern_name
        
        # Check for mixed malicious queries (safe topic + dangerous topic)
        safe_keywords = ['journvax', 'medication', 'side effect', 'dosage', 'usage']
        danger_keywords = ['bomb', 'weapon', 'explosive', 'sexy', 'erotic', 'kill', 'harm']
        
        has_safe = any(keyword in query_lower for keyword in safe_keywords)
        has_danger = any(keyword in query_lower for keyword in danger_keywords)
        
        if has_safe and has_danger:
            logger.warning("Mixed malicious query detected")
            return ThreatType.MIXED_MALICIOUS, "mixed_query"
        
        return ThreatType.NONE, ""
    
    def check_response_safety(self, response: str) -> Tuple[bool, str]:
        """
        Check response for safety issues
        Returns: (is_safe, reason)
        """
        response_lower = response.lower()
        
        for pattern, pattern_name in self.response_safety_patterns:
            if re.search(pattern, response_lower):
                logger.warning(f"Response safety issue: {pattern_name}")
                return False, f"Contains {pattern_name}"
        
        return True, "No safety issues"
    
    def calculate_grounding_score(self, response: str, context: str) -> float:
        """
        Calculate semantic similarity with better handling
        """
        if not context or not self.embedding_model:
            return 0.0
        
        try:
            response_embedding = self.embedding_model.encode(
                response, convert_to_tensor=False, show_progress_bar=False
            )
            context_embedding = self.embedding_model.encode(
                context, convert_to_tensor=False, show_progress_bar=False
            )
            
            similarity = float(
                np.dot(response_embedding, context_embedding) / 
                (np.linalg.norm(response_embedding) * np.linalg.norm(context_embedding))
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            return 0.5
    
    def enhance_response_transparency(self, response: str, context: str) -> str:
        """
        Add transparency markers to responses
        """
        # Check if response contains medical information
        medical_terms = ['side effect', 'dosage', 'symptom', 'treatment', 'medication']
        contains_medical = any(term in response.lower() for term in medical_terms)
        
        if contains_medical and context:
            # Add disclaimer if not already present
            disclaimers = [
                "according to", "based on", "documentation states", 
                "information provided", "not a complete list"
            ]
            has_disclaimer = any(disc in response.lower() for disc in disclaimers)
            
            if not has_disclaimer:
                response = response.rstrip('.')
                response += ". Please note this information is based on available documentation and may not be complete. Always consult your healthcare provider for personalized advice."
        
        return response
    
    async def validate_query(self, query: str) -> Optional[ValidationDecision]:
        """
        Validate query BEFORE processing (early detection)
        Returns ValidationDecision if query should be blocked, None if safe
        """
        if not self.enabled:
            return None
        
        # Check medical safety FIRST (highest priority)
        if self.medical_detector:
            med_type, pattern, is_emergency = self.medical_detector.detect_medical_request(query)
            if med_type != ThreatType.NONE:
                logger.warning(f"Medical safety issue detected: {pattern}")
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=self.medical_detector.get_safe_response(med_type, is_emergency),
                    reasoning=f"Medical safety: {pattern}",
                    confidence=0.98,
                    threat_type=ThreatType.UNSAFE_MEDICAL,
                    should_log=True
                )
        
        # Check for off-topic but harmless requests (before threat detection)
        if self.off_topic_handler:
            off_topic_response = self.off_topic_handler.generate_response(query)
            if off_topic_response:
                logger.info("Off-topic request handled gracefully")
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=off_topic_response,
                    reasoning="Off-topic request",
                    confidence=0.95,
                    threat_type=ThreatType.OFF_TOPIC,
                    should_log=False  # Don't log harmless off-topic
                )
        
        # Then detect other threats
        threat_type, pattern_name = self.detect_query_threats(query)
        
        if threat_type != ThreatType.NONE:
            # Return appropriate response for threat type
            return ValidationDecision(
                result=ValidationResult.REDIRECT,
                final_response=self.threat_responses[threat_type],
                reasoning=f"Query contains {threat_type.value}: {pattern_name}",
                confidence=0.95,
                threat_type=threat_type,
                should_log=True  # Log for security monitoring
            )
        
        return None  # Query is safe to process
    
    async def validate_response(
        self,
        response: str,
        context: str = "",
        query: str = "",
        **kwargs
    ) -> ValidationDecision:
        """
        Validate response AFTER generation with compliance enforcement
        """
        if not self.enabled:
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation disabled",
                confidence=1.0
            )
        
        try:
            # First, check if query had threats we missed
            threat_type, _ = self.detect_query_threats(query)
            if threat_type != ThreatType.NONE:
                # Should have been caught earlier, but block anyway
                # Use approved templates for refusals
                if self.compliance_validator:
                    final_response = self.compliance_validator.enforce_compliance(
                        self.threat_responses[threat_type],
                        query,
                        is_refusal=True
                    )
                else:
                    final_response = self.threat_responses[threat_type]
                
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=final_response,
                    reasoning=f"Late detection of {threat_type.value}",
                    confidence=0.9,
                    threat_type=threat_type,
                    should_log=True
                )
            
            # Check response safety
            is_safe, safety_reason = self.check_response_safety(response)
            if not is_safe:
                safe_response = "I cannot provide that information as it may be unsafe. Please consult with a healthcare professional for medical advice."
                
                # Enforce compliance on the safe response
                if self.compliance_validator:
                    safe_response = self.compliance_validator.enforce_compliance(
                        safe_response,
                        query,
                        is_refusal=True
                    )
                
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=safe_response,
                    reasoning=f"Response safety check failed: {safety_reason}",
                    confidence=0.95
                )
            
            # Check for off-topic or unhelpful responses
            response_lower = response.lower()
            off_topic_indicators = [
                "i don't have that information",
                "i don't have information about that",
                "not in the documentation",
                "i cannot help with that",
                "outside my scope",
            ]
            
            if any(indicator in response_lower for indicator in off_topic_indicators):
                final_response = "I don't have specific information about that in the Journvax documentation. Could you please ask something specific about Journvax?"
                
                # Enforce compliance
                if self.compliance_validator:
                    final_response = self.compliance_validator.enforce_compliance(
                        final_response,
                        query,
                        is_refusal=False
                    )
                
                return ValidationDecision(
                    result=ValidationResult.APPROVED,
                    final_response=final_response,
                    reasoning="Off-topic response detected",
                    confidence=0.9,
                    threat_type=ThreatType.OFF_TOPIC
                )
            
            # Check grounding if context provided
            if context and len(context) > 100:
                grounding_score = self.calculate_grounding_score(response, context)
                
                if grounding_score < 0.2:  # Very poor grounding
                    logger.warning(f"Poor grounding: {grounding_score:.2f}")
                    final_response = "I don't have sufficient information to answer that question accurately. Please ask about specific aspects of Journvax that I can help with."
                    
                    # Enforce compliance
                    if self.compliance_validator:
                        final_response = self.compliance_validator.enforce_compliance(
                            final_response,
                            query,
                            is_refusal=False
                        )
                    
                    return ValidationDecision(
                        result=ValidationResult.APPROVED,
                        final_response=final_response,
                        reasoning=f"Poor grounding score: {grounding_score:.2f}",
                        confidence=0.7
                    )
            
            # COMPLIANCE ENFORCEMENT - Apply to all approved responses
            if self.compliance_validator:
                # Determine if this is a refusal based on content
                is_refusal = response_lower.startswith(("i cannot", "i will not", "i do not"))
                
                # Enforce compliance rules (now includes final safety)
                enhanced_response = self.compliance_validator.enforce_compliance(
                    response,
                    query,
                    is_refusal=is_refusal
                )
            else:
                enhanced_response = self.enhance_response_transparency(response, context)
            
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=enhanced_response,
                reasoning="Passed all safety and compliance checks",
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation error, approved by default",
                confidence=0.5
            )

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create singleton instance
enhanced_guard = EnhancedGuard()

# Legacy support
simple_guard = enhanced_guard
persona_validator = enhanced_guard

def evaluate_response(
    context: str,
    user_question: str,
    assistant_response: str,
    conversation_history: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Legacy function for backward compatibility"""
    import asyncio
    
    async def validate():
        decision = await enhanced_guard.validate_response(
            response=assistant_response,
            context=context,
            query=user_question
        )
        return decision
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, validate())
            decision = future.result()
    else:
        decision = loop.run_until_complete(validate())
    
    is_approved = decision.result == ValidationResult.APPROVED
    return is_approved, decision.final_response, decision.reasoning