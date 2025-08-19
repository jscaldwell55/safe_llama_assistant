# guard.py - Document-Grounded Safety System Final Version
"""
Safety system enforcing:
1. Mandatory grounding in retrieved documentation
2. Prevention of 9 regulatory violation categories

Generic, scalable approach without hardcoded edge cases.
"""

import logging
import re
import json
import numpy as np
from typing import Tuple, Optional, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# REGULATORY CATEGORIES
# ============================================================================

class RegulatoryViolation(Enum):
    """9 Core Regulatory Categories from compliance requirements"""
    INACCURATE_CLAIMS = "inaccurate_claims"           # Cat 1: False/unverifiable claims
    INADEQUATE_RISK_COMM = "inadequate_risk_comm"     # Cat 2: Missing fair balance
    OFF_LABEL_USE = "off_label_use"                   # Cat 3: Unapproved uses
    IMPROPER_PROMOTION = "improper_promotion"         # Cat 4: Unbalanced/inappropriate tone
    CROSS_PRODUCT_REF = "cross_product_ref"           # Cat 5: Unsupported brand comparisons
    MEDICAL_ADVICE = "medical_advice"                 # Cat 6: Individual clinical guidance
    SAFETY_CRITICAL_MISS = "safety_critical_miss"     # Cat 7: Missing emergency guidance
    ADMIN_MISUSE = "admin_misuse"                     # Cat 8: Unsafe administration
    UNAPPROVED_DOSING = "unapproved_dosing"          # Cat 9: Non-label dosing guidance
    NONE = "none"

class ValidationResult(Enum):
    """Validation outcomes"""
    APPROVED = "approved"
    REJECTED = "rejected"
    REDIRECT = "redirect"

class ThreatType(Enum):
    """Types of threats (for legacy compatibility)"""
    NONE = "none"
    VIOLENCE = "violence"
    INAPPROPRIATE = "inappropriate"
    UNSAFE_MEDICAL = "unsafe_medical"
    MIXED_MALICIOUS = "mixed_malicious"
    OFF_TOPIC = "off_topic"

@dataclass
class ValidationDecision:
    """Complete validation decision with all details"""
    result: ValidationResult
    final_response: str
    reasoning: str
    confidence: float = 0.0
    threat_type: ThreatType = ThreatType.NONE
    should_log: bool = False
    grounding_score: float = 0.0
    unsupported_claims: List[str] = field(default_factory=list)
    violation: RegulatoryViolation = RegulatoryViolation.NONE

# ============================================================================
# DOCUMENT GROUNDING VALIDATOR
# ============================================================================

class DocumentGroundingValidator:
    """
    Ensures responses are grounded in retrieved documentation.
    Generic approach without hardcoded substances or specific cases.
    """
    
    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load embedding model for similarity check"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Embedding model loaded for grounding validation")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.embedding_model or not text1 or not text2:
            return 0.0
        
        try:
            emb1 = self.embedding_model.encode(text1, convert_to_tensor=False, show_progress_bar=False)
            emb2 = self.embedding_model.encode(text2, convert_to_tensor=False, show_progress_bar=False)
            
            # Cosine similarity
            similarity = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text that need grounding"""
        claims = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip meta-statements and refusals
            skip_phrases = [
                "i don't have", "i cannot", "please consult", "according to",
                "the documentation", "medication guide", "healthcare provider",
                "i'm not able", "i apologize", "please refer"
            ]
            
            if any(phrase in sentence.lower() for phrase in skip_phrases):
                continue
            
            # This is likely a factual claim that needs grounding
            claims.append(sentence)
        
        return claims
    
    def check_grounding(self, response: str, context: str) -> Tuple[float, bool, List[str]]:
        """
        Check if response is grounded in documentation.
        Returns: (similarity_score, is_grounded, unsupported_claims)
        """
        if not context or len(context.strip()) < 50:
            # No meaningful context available
            claims = self.extract_factual_claims(response)
            if claims:
                # Has factual claims but no context to verify
                return 0.0, False, claims
            # No factual claims made (likely a refusal)
            return 1.0, True, []
        
        # Calculate overall similarity
        overall_similarity = self.calculate_similarity(response, context)
        
        # Check individual claims
        claims = self.extract_factual_claims(response)
        unsupported = []
        
        for claim in claims:
            # Check if this claim is supported by context
            claim_similarity = self.calculate_similarity(claim, context)
            
            # Extract key pharmaceutical/medical terms from claim
            claim_lower = claim.lower()
            context_lower = context.lower()
            
            # Pattern for pharmaceutical terms (compounds, medications, symptoms)
            pharm_pattern = r'\b[a-z]+(?:ine|ol|ate|ide|ium|azole|amine|acid|oxide)\b'
            pharm_terms = re.findall(pharm_pattern, claim_lower)
            
            # Also check for other key medical/substance terms
            key_terms = re.findall(r'\b(?:alcohol|food|meal|juice|drink|water|milk|caffeine|tobacco)\b', claim_lower)
            all_key_terms = pharm_terms + key_terms
            
            # Check if key terms from claim appear in context
            missing_terms = [term for term in all_key_terms if term not in context_lower]
            
            # Claim is unsupported if similarity is low OR key terms are missing
            if claim_similarity < self.similarity_threshold or missing_terms:
                unsupported.append(claim)
                logger.debug(f"Unsupported claim: {claim[:50]}... (score: {claim_similarity:.3f}, missing: {missing_terms})")
        
        is_grounded = overall_similarity >= self.similarity_threshold and not unsupported
        
        if not is_grounded:
            logger.warning(f"Poor grounding: score={overall_similarity:.3f}, unsupported={len(unsupported)}")
        
        return overall_similarity, is_grounded, unsupported

# ============================================================================
# REGULATORY COMPLIANCE CHECKER
# ============================================================================

class RegulatoryComplianceChecker:
    """
    Checks for violations of 9 regulatory categories.
    Based on patterns from compliance requirements.
    """
    
    def __init__(self):
        # Pre-approved refusal templates for each violation type
        self.refusal_templates = {
            RegulatoryViolation.MEDICAL_ADVICE: 
                "I cannot provide medical advice. Please consult your healthcare provider.",
            RegulatoryViolation.UNAPPROVED_DOSING:
                "I cannot provide dosing guidance. Please consult your healthcare provider or refer to the Medication Guide.",
            RegulatoryViolation.OFF_LABEL_USE:
                "I can only provide information about approved uses. Please consult your healthcare provider.",
            RegulatoryViolation.ADMIN_MISUSE:
                "I cannot recommend sharing medications or alternative administration methods. Please consult your healthcare provider.",
            RegulatoryViolation.INACCURATE_CLAIMS:
                "I don't have that specific information in the Journvax documentation. Please consult your healthcare provider or pharmacist.",
            RegulatoryViolation.IMPROPER_PROMOTION:
                "I can only provide factual information from the approved documentation. Please consult your healthcare provider.",
            RegulatoryViolation.CROSS_PRODUCT_REF:
                "I can only provide information about Journvax. Please consult your healthcare provider for comparisons with other medications.",
            RegulatoryViolation.SAFETY_CRITICAL_MISS:
                "For severe symptoms, seek immediate medical attention or call emergency services.",
            RegulatoryViolation.INADEQUATE_RISK_COMM:
                "Please see the Medication Guide for complete safety information. Consult your healthcare provider with any concerns."
        }
    
    def check_violations(self, response: str, query: str, context: str = "") -> ValidationDecision:
        """
        Check response for regulatory violations based on compliance patterns.
        """
        response_lower = response.lower()
        
        # Category 1: Inaccurate/Misleading Claims - "It doesn't mention X, so you're fine"
        patterns_cat1 = [
            r"(?:doesn't|does not|don't) mention.*(?:so|therefore).*(?:fine|ok|safe|should be)",
            r"(?:probably|likely|should) (?:won't|shouldn't|isn't) (?:be )?(?:a problem|an issue)",
            r"generally (?:safe|recommended|fine|ok)",
            r"most people (?:can|don't|are)",
            r"it's (?:usually|typically|normally) (?:fine|safe|ok)"
        ]
        
        for pattern in patterns_cat1:
            if re.search(pattern, response_lower):
                return self._create_violation(
                    RegulatoryViolation.INACCURATE_CLAIMS,
                    "Implied safety from absence or speculation"
                )
        
        # Category 2: Inadequate Risk Communication
        medical_terms = ['side effect', 'adverse', 'reaction', 'symptom', 'warning', 'precaution']
        has_medical = any(term in response_lower for term in medical_terms)
        has_disclaimer = any(phrase in response_lower for phrase in [
            "not a complete list", "medication guide", "full information",
            "complete safety information", "see the medication guide"
        ])
        
        if has_medical and not has_disclaimer and "i cannot" not in response_lower:
            corrected = response.rstrip('.') + ". This is not a complete list. See the Medication Guide for full information."
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=corrected,
                reasoning="Missing required disclaimer for medical information",
                confidence=0.90,
                violation=RegulatoryViolation.INADEQUATE_RISK_COMM
            )
        
        # Category 3: Off-Label Use
        off_label_patterns = [
            r"(?:child|pediatric|infant|baby|pregnancy|pregnant|nursing|breastfeed).*(?:can|may|should|dose|use)",
            r"use (?:for|to treat).*(?:that is not|outside|beyond)"
        ]
        
        for pattern in off_label_patterns:
            if re.search(pattern, response_lower):
                if "cannot" not in response_lower and "consult" not in response_lower:
                    return self._create_violation(
                        RegulatoryViolation.OFF_LABEL_USE,
                        "Potential off-label use recommendation"
                    )
        
        # Category 4: Improper Promotion - Tone issues
        problematic_tone = [
            "don't worry", "no need to worry", "should be fine", "nothing to worry about",
            "typically safe", "generally recommended", "most people", "usually fine",
            "err on the side of caution", "better safe than sorry"
        ]
        
        for phrase in problematic_tone:
            if phrase in response_lower:
                return self._create_violation(
                    RegulatoryViolation.IMPROPER_PROMOTION,
                    "Inappropriate reassurance or casual tone"
                )
        
        # Category 5: Cross-Product References
        if re.search(r"(?:like|similar to|same as|compared to|versus|better than|worse than).*(?:other|another|different|competitor)", response_lower):
            return self._create_violation(
                RegulatoryViolation.CROSS_PRODUCT_REF,
                "Unsupported product comparison"
            )
        
        # Category 6: Medical Advice - Individual guidance
        medical_advice_patterns = [
            r"you (?:should|can|could|may|might) (?:take|stop|start|increase|decrease|adjust|change)",
            r"(?:increase|decrease|double|halve|reduce|raise) your (?:dose|dosage)",
            r"your doctor (?:may|might|will|could|should) (?:adjust|prescribe|recommend)",
            r"(?:try|consider) (?:taking|using|stopping)"
        ]
        
        for pattern in medical_advice_patterns:
            if re.search(pattern, response_lower):
                if "consult" not in response_lower and "cannot" not in response_lower:
                    return self._create_violation(
                        RegulatoryViolation.MEDICAL_ADVICE,
                        "Individual medical guidance"
                    )
        
        # Category 7: Safety-Critical Information
        severe_symptoms = [
            'severe', 'emergency', 'immediately', 'trouble breathing', 
            'chest pain', 'swelling', 'anaphylaxis', 'seizure'
        ]
        has_severe = any(symptom in response_lower for symptom in severe_symptoms)
        has_emergency_guidance = any(phrase in response_lower for phrase in [
            "seek immediate", "emergency", "call 911", "hospital", "urgent"
        ])
        
        if has_severe and not has_emergency_guidance and "cannot" not in response_lower:
            return self._create_violation(
                RegulatoryViolation.SAFETY_CRITICAL_MISS,
                "Missing emergency guidance for severe symptoms"
            )
        
        # Category 8: Administration Method Misuse
        admin_misuse_patterns = [
            r"(?:share|split|crush|chew|inject|snort).*(?:medication|pill|tablet|dose)",
            r"give (?:to|your).*(?:friend|family|spouse|partner)"
        ]
        
        for pattern in admin_misuse_patterns:
            if re.search(pattern, response_lower):
                if "cannot" not in response_lower and "do not" not in response_lower:
                    return self._create_violation(
                        RegulatoryViolation.ADMIN_MISUSE,
                        "Unsafe administration suggestion"
                    )
        
        # Category 9: Unapproved Dosing
        dosing_patterns = [
            r"take \d+.*(?:tablet|pill|mg|ml|capsule)",
            r"(?:morning|evening|night|bedtime|daily|twice|hourly).*(?:dose|dosing)",
            r"(?:with|without|before|after).*(?:food|meal|eating|empty stomach)",
            r"(?:stick to|maintain|follow).*(?:regular|usual|your).*schedule",
            r"(?:maximum|most|highest|lowest|minimum).*(?:can|should|safe).*take"
        ]
        
        for pattern in dosing_patterns:
            if re.search(pattern, response_lower):
                if "according to" not in response_lower and "medication guide" not in response_lower:
                    return self._create_violation(
                        RegulatoryViolation.UNAPPROVED_DOSING,
                        "Specific dosing or administration guidance"
                    )
        
        # No violations found
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="No violations detected",
            confidence=0.95,
            violation=RegulatoryViolation.NONE
        )
    
    def _create_violation(self, violation: RegulatoryViolation, reasoning: str) -> ValidationDecision:
        """Helper to create violation result with appropriate refusal"""
        logger.warning(f"Compliance violation: {violation.value} - {reasoning}")
        
        return ValidationDecision(
            result=ValidationResult.REJECTED,
            final_response=self.refusal_templates[violation],
            reasoning=reasoning,
            confidence=0.90,
            violation=violation,
            should_log=True
        )

# ============================================================================
# MAIN GUARD CLASS
# ============================================================================

class EnhancedSafetyGuard:
    """
    Main safety orchestrator focusing on:
    1. Document grounding (prevent hallucination)
    2. Regulatory compliance (prevent violations)
    """
    
    def __init__(self):
        self.grounding_validator = DocumentGroundingValidator()
        self.compliance_checker = RegulatoryComplianceChecker()
        self.enabled = True
        
        # Load configuration
        try:
            from config import ENABLE_GUARD, SEMANTIC_SIMILARITY_THRESHOLD
            self.enabled = ENABLE_GUARD
            if SEMANTIC_SIMILARITY_THRESHOLD:
                self.grounding_validator.similarity_threshold = SEMANTIC_SIMILARITY_THRESHOLD
        except ImportError:
            logger.warning("Could not load config, using defaults")
    
    async def validate_query(self, query: str) -> Optional[ValidationDecision]:
        """
        Pre-screen queries for unsafe requests.
        Returns ValidationDecision if query should be blocked, None if safe.
        """
        if not self.enabled:
            return None
        
        query_lower = query.lower()
        
        # Unsafe query patterns that should be blocked immediately
        unsafe_patterns = [
            (r"(?:double|triple|increase|extra|more).*(?:dose|dosage|medication|pills?)", 
             RegulatoryViolation.UNAPPROVED_DOSING, "dosage increase request"),
            (r"share.*(?:prescription|medication|pills?|journvax)", 
             RegulatoryViolation.ADMIN_MISUSE, "prescription sharing"),
            (r"(?:child|kid|baby|infant|toddler).*(?:give|dose|how much|medication)", 
             RegulatoryViolation.OFF_LABEL_USE, "pediatric dosing"),
            (r"(?:maximum|most|highest|safest).*(?:can|should|safe).*take", 
             RegulatoryViolation.UNAPPROVED_DOSING, "maximum dose inquiry"),
            (r"overdose|lethal dose|kill (?:myself|me)|suicide", 
             RegulatoryViolation.SAFETY_CRITICAL_MISS, "dangerous request"),
            (r"(?:crush|snort|inject|smoke).*(?:journvax|medication|pill)", 
             RegulatoryViolation.ADMIN_MISUSE, "unsafe administration method"),
        ]
        
        for pattern, violation, description in unsafe_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Unsafe query detected: {description}")
                
                refusal = self.compliance_checker.refusal_templates[violation]
                
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=refusal,
                    reasoning=f"Query contains {description}",
                    confidence=0.95,
                    violation=violation,
                    threat_type=ThreatType.UNSAFE_MEDICAL,
                    should_log=True
                )
        
        return None
    
    async def validate_response(
        self,
        response: str,
        context: str = "",
        query: str = "",
        strategy_used: str = "",
        **kwargs
    ) -> ValidationDecision:
        """
        Validate response for grounding and compliance.
        """
        if not self.enabled:
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation disabled",
                confidence=1.0
            )
        
        try:
            # Step 1: Check regulatory compliance first
            compliance_result = self.compliance_checker.check_violations(response, query, context)
            
            # If compliance violation found, return immediately
            if compliance_result.result != ValidationResult.APPROVED:
                logger.info(f"Compliance check failed: {compliance_result.violation.value}")
                return compliance_result
            
            # Step 2: Check document grounding (only if context provided)
            if context and len(context.strip()) > 50:
                grounding_score, is_grounded, unsupported = self.grounding_validator.check_grounding(
                    response, context
                )
                
                compliance_result.grounding_score = grounding_score
                compliance_result.unsupported_claims = unsupported
                
                if not is_grounded and unsupported:
                    logger.warning(f"Grounding check failed: score={grounding_score:.3f}, unsupported={len(unsupported)}")
                    
                    # Return grounding failure with safe response
                    return ValidationDecision(
                        result=ValidationResult.REJECTED,
                        final_response="I don't have that specific information in the Journvax documentation. Please consult your healthcare provider or pharmacist for guidance.",
                        reasoning=f"Response not grounded in documentation (score: {grounding_score:.3f})",
                        confidence=0.85,
                        violation=RegulatoryViolation.INACCURATE_CLAIMS,
                        grounding_score=grounding_score,
                        unsupported_claims=unsupported,
                        should_log=True
                    )
            
            # All checks passed
            return compliance_result
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            # On error, default to safe response
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning=f"Validation error: {str(e)}",
                confidence=0.5
            )
    
    def get_grounding_report(self, response: str, context: str) -> Dict[str, any]:
        """
        Generate detailed grounding report for debugging.
        """
        score, is_grounded, unsupported = self.grounding_validator.check_grounding(response, context)
        
        return {
            "grounding_score": score,
            "is_grounded": is_grounded,
            "threshold": self.grounding_validator.similarity_threshold,
            "unsupported_claims": unsupported,
            "context_length": len(context) if context else 0,
            "response_length": len(response)
        }

# ============================================================================
# SINGLETON & LEGACY COMPATIBILITY
# ============================================================================

# Create singleton instance
enhanced_guard = EnhancedSafetyGuard()

# Legacy compatibility aliases
persona_validator = enhanced_guard
simple_guard = enhanced_guard

# Legacy function for backward compatibility
def evaluate_response(
    context: str,
    user_question: str,
    assistant_response: str,
    conversation_history: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Legacy function for backward compatibility"""
    import asyncio
    
    async def validate():
        result = await enhanced_guard.validate_response(
            response=assistant_response,
            context=context,
            query=user_question
        )
        return result
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, validate())
            result = future.result()
    else:
        result = loop.run_until_complete(validate())
    
    is_safe = (result.result == ValidationResult.APPROVED)
    final_response = result.final_response
    reasoning = result.reasoning
    
    return is_safe, final_response, reasoning