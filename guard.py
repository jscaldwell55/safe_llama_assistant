# guard.py - Complete Safety System with All Fixes
"""
Comprehensive safety system enforcing:
1. Strict document grounding (no external knowledge)
2. Six critical regulatory violations
3. Violence/harm/illegal activity detection
4. No creative content generation
5. Proper handling of legitimate information requests
"""

import logging
import re
import json
import numpy as np
import asyncio
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class RegulatoryViolation(Enum):
    """Critical Regulatory Categories"""
    OFF_LABEL_USE = "off_label_use"                          # Unapproved uses/populations
    MEDICAL_ADVICE = "medical_advice"                        # Practicing medicine
    CROSS_PRODUCT_REF = "cross_product_ref"                  # Other drug references
    ADMIN_MISUSE = "admin_misuse"                           # Dangerous administration
    INACCURATE_CLAIMS = "inaccurate_claims"                 # False/unsupported claims
    INADEQUATE_RISK_COMM = "inadequate_risk_comm"           # Missing safety warnings
    
    # Additional safety categories
    VIOLENCE_HARM = "violence_harm"                         # Violence or self-harm
    ILLEGAL_ACTIVITY = "illegal_activity"                   # Illegal drugs/activities
    CREATIVE_CONTENT = "creative_content"                   # Stories/poems/fiction
    OFF_TOPIC = "off_topic"                                # Non-Journvax topics
    INSUFFICIENT_GROUNDING = "insufficient_grounding"       # Not from documentation
    NONE = "none"

class ValidationResult(Enum):
    """Validation outcomes"""
    APPROVED = "approved"
    REJECTED = "rejected"
    REDIRECT = "redirect"

@dataclass
class ValidationDecision:
    """Unified validation decision"""
    result: ValidationResult
    final_response: str
    reasoning: str
    confidence: float = 0.0
    grounding_score: float = 0.0
    unsupported_claims: List[str] = field(default_factory=list)
    violation: RegulatoryViolation = RegulatoryViolation.NONE
    should_log: bool = False

# ============================================================================
# STANDARD RESPONSES
# ============================================================================

STANDARD_REFUSAL = "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"
SAFETY_REFUSAL = "I'm sorry, I cannot discuss that. Would you like to talk about something else?"

# ============================================================================
# HYBRID SAFETY GUARD
# ============================================================================

class HybridSafetyGuard:
    """
    Unified safety system with strict document grounding and violation detection
    """
    
    def __init__(self):
        self.enabled = True
        self.similarity_threshold = 0.50  # Raised for stricter grounding
        self.embedding_model = None
        
        # Load configuration
        try:
            from config import ENABLE_GUARD, SEMANTIC_SIMILARITY_THRESHOLD
            self.enabled = ENABLE_GUARD
            # Override with stricter threshold
            self.similarity_threshold = max(0.50, SEMANTIC_SIMILARITY_THRESHOLD)
            logger.info(f"Guard enabled: {self.enabled}, threshold: {self.similarity_threshold}")
        except ImportError:
            logger.warning("Using default configuration")
        
        self._load_models()
    
    def _load_models(self):
        """Load embedding model for grounding checks"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Embedding model loaded for grounding")
        except Exception as e:
            logger.error(f"Could not load embedding model: {e}")
    
    # ========================================================================
    # QUERY VALIDATION - FIXED WITH LEGITIMATE INFO REQUEST HANDLING
    # ========================================================================
    
    async def validate_query(self, query: str) -> Optional[ValidationDecision]:
        """Pre-screen queries for safety violations"""
        if not self.enabled:
            return None
        
        query_lower = query.lower()
        
        # FIRST: Check if this is a legitimate information request about interactions/contraindications
        legitimate_info_patterns = [
            r'what.*(?:interact|contraindication|can.*not.*take.*with|should.*avoid)',
            r'(?:what|which).*(?:food|drug|medication).*(?:avoid|interact)',
            r'dangerous.*(?:combination|interaction)',
            r'who should not take',
            r'what.*not.*(?:take|mix|combine).*with',
            r'drug interaction',
            r'contraindication',
        ]
        
        for pattern in legitimate_info_patterns:
            if re.search(pattern, query_lower):
                # This is asking for factual information - allow it
                logger.info("Legitimate information request about interactions/contraindications - allowing")
                return None
        
        # CRITICAL: Block violence/harm/suicide
        violence_patterns = [
            r'\b(kill|murder|suicide|harm|hurt|poison|overdose|od|lethal|fatal|die|death)\b',
            r'end my life|take my life|hurt myself|harm myself',
            r'how much.*(?:kill|die|lethal|fatal)',
        ]
        
        for pattern in violence_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"CRITICAL: Violence/harm query blocked")
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=SAFETY_REFUSAL,
                    reasoning="Violence/harm content detected",
                    violation=RegulatoryViolation.VIOLENCE_HARM,
                    confidence=1.0,
                    should_log=True
                )
        
        # Block illegal drugs (including "speedball")
        illegal_terms = [
            'cocaine', 'heroin', 'meth', 'crack', 'ecstasy', 'mdma', 'lsd',
            'speedball', 'mushrooms', 'pcp', 'weed', 'marijuana', 'cannabis',
            'fentanyl', 'opioid', 'opiate', 'narcotic'
        ]
        
        # Check context - allow if asking about legitimate drug interactions
        if not re.search(r'(?:interact|can.*take|mix|combine)', query_lower):
            for term in illegal_terms:
                if term in query_lower:
                    logger.warning(f"Illegal drug reference blocked: {term}")
                    return ValidationDecision(
                        result=ValidationResult.REDIRECT,
                        final_response=SAFETY_REFUSAL,
                        reasoning=f"Illegal drug reference: {term}",
                        violation=RegulatoryViolation.ILLEGAL_ACTIVITY,
                        confidence=1.0,
                        should_log=True
                    )
        
        # Block creative content requests
        creative_patterns = [
            r'\b(story|stories|poem|poetry|fiction|narrative|tale|essay|script)\b',
            r'tell me.*story|write.*story|create.*story',
            r'in the context of.*story',
            r'pretend|imagine|roleplay|act as',
        ]
        
        for pattern in creative_patterns:
            if re.search(pattern, query_lower):
                logger.warning("Creative content request blocked")
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=STANDARD_REFUSAL,
                    reasoning="Creative content request",
                    violation=RegulatoryViolation.CREATIVE_CONTENT,
                    confidence=1.0
                )
        
        # Block clearly off-topic
        if not any(term in query_lower for term in ['journvax', 'medication', 'drug', 'side', 'effect', 'dose', 'interact', 'take']):
            # Check if it's about something unrelated
            off_topic_terms = ['gravity', 'physics', 'weather', 'recipe', 'math', 'history', 'geography']
            if any(term in query_lower for term in off_topic_terms):
                logger.warning("Off-topic query blocked")
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=STANDARD_REFUSAL,
                    reasoning="Off-topic query",
                    violation=RegulatoryViolation.OFF_TOPIC,
                    confidence=1.0
                )
        
        # Medical advice patterns - MORE SPECIFIC to avoid false positives
        medical_patterns = [
            # Personal dosing questions
            (r"(?:how much|what dose).*(?:should i take|do i need|for me)", RegulatoryViolation.MEDICAL_ADVICE),
            
            # Personal medical decisions with "I" statements
            (r"(?:i take|i'm taking|i use).*(?:can i|should i|is it safe)", RegulatoryViolation.MEDICAL_ADVICE),
            (r"(?:should i|can i personally|is it safe for me to).*(?:take|use|start|stop)", RegulatoryViolation.MEDICAL_ADVICE),
            
            # Personal conditions
            (r"(?:i have|i'm|i am|my).*(?:condition|disease|pregnant|breastfeed|allerg)", RegulatoryViolation.MEDICAL_ADVICE),
        ]
        
        for pattern, violation in medical_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Medical advice query blocked")
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=SAFETY_REFUSAL,
                    reasoning="Medical advice request",
                    violation=violation,
                    confidence=0.95
                )
        
        # Off-label use patterns
        off_label_patterns = [
            (r"(?:child|kid|baby|infant|pediatric|teenager|minor)", RegulatoryViolation.OFF_LABEL_USE),
            (r"(?:pregnant|pregnancy|breastfeed|nursing)", RegulatoryViolation.OFF_LABEL_USE),
        ]
        
        # Only block if it's asking about personal use, not general information
        if re.search(r"(?:my|can i|should i|for my)", query_lower):
            for pattern, violation in off_label_patterns:
                if re.search(pattern, query_lower):
                    logger.warning("Off-label use query blocked")
                    return ValidationDecision(
                        result=ValidationResult.REDIRECT,
                        final_response=SAFETY_REFUSAL,
                        reasoning="Off-label use query",
                        violation=violation,
                        confidence=0.95
                    )
        
        # Administration misuse
        admin_patterns = [
            (r"(?:crush|snort|inject|smoke|iv|intravenous)", RegulatoryViolation.ADMIN_MISUSE),
            (r"(?:share|give to someone|sell)", RegulatoryViolation.ADMIN_MISUSE),
        ]
        
        for pattern, violation in admin_patterns:
            if re.search(pattern, query_lower):
                logger.warning("Administration misuse query blocked")
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=SAFETY_REFUSAL,
                    reasoning="Administration misuse",
                    violation=violation,
                    confidence=0.95
                )
        
        # Alcohol questions - allow general info, block personal use
        if 'alcohol' in query_lower or 'drink' in query_lower:
            if re.search(r"(?:can i|should i|if i)", query_lower):
                return ValidationDecision(
                    result=ValidationResult.REDIRECT,
                    final_response=SAFETY_REFUSAL,
                    reasoning="Personal alcohol interaction query",
                    violation=RegulatoryViolation.MEDICAL_ADVICE,
                    confidence=0.95
                )
        
        return None
    
    # ========================================================================
    # RESPONSE VALIDATION - STRICT GROUNDING
    # ========================================================================
    
    async def validate_response(
        self,
        response: str,
        context: str = "",
        query: str = "",
        **kwargs
    ) -> ValidationDecision:
        """Validate response with strict grounding and safety checks"""
        if not self.enabled:
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation disabled",
                confidence=1.0
            )
        
        response_lower = response.lower()
        
        # Skip validation for standard refusals
        if response_lower.startswith(("i'm sorry", "i cannot", "i don't have", "i don't seem")):
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Standard refusal",
                confidence=1.0
            )
        
        # CRITICAL: Block any creative content in response
        if any(word in response_lower for word in ['story', 'tale', 'once upon', 'narrative', 'fiction']):
            logger.warning("Creative content in response blocked")
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=STANDARD_REFUSAL,
                reasoning="Creative content detected",
                violation=RegulatoryViolation.CREATIVE_CONTENT,
                confidence=1.0
            )
        
        # Check for off-topic content
        off_topic_terms = ['gravity', 'physics', 'weather', 'mathematics', 'history', 'geography']
        if any(term in response_lower for term in off_topic_terms):
            if 'journvax' not in response_lower:
                logger.warning("Off-topic response blocked")
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=STANDARD_REFUSAL,
                    reasoning="Off-topic content",
                    violation=RegulatoryViolation.OFF_TOPIC,
                    confidence=1.0
                )
        
        # STRICT GROUNDING CHECK
        if context and len(context) > 50:
            grounding_result = await self._check_strict_grounding(response, context)
            if grounding_result.result != ValidationResult.APPROVED:
                return grounding_result
        elif not context:
            # No context means no documentation was retrieved
            logger.warning("No context available - response must be rejected")
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=STANDARD_REFUSAL,
                reasoning="No documentation context available",
                violation=RegulatoryViolation.INSUFFICIENT_GROUNDING,
                confidence=1.0
            )
        
        # Check compliance patterns
        compliance_result = self._check_compliance_patterns(response, query)
        if compliance_result.result != ValidationResult.APPROVED:
            return compliance_result
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="All checks passed",
            confidence=0.95
        )
    
    async def _check_strict_grounding(self, response: str, context: str) -> ValidationDecision:
        """Strict grounding check - response must be based on context"""
        if not self.embedding_model:
            # Without embedding model, be conservative
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=STANDARD_REFUSAL,
                reasoning="Cannot verify grounding",
                violation=RegulatoryViolation.INSUFFICIENT_GROUNDING,
                confidence=0.5
            )
        
        try:
            # Calculate similarity
            resp_emb = self.embedding_model.encode(response, show_progress_bar=False)
            ctx_emb = self.embedding_model.encode(context, show_progress_bar=False)
            similarity = float(np.dot(resp_emb, ctx_emb) / (np.linalg.norm(resp_emb) * np.linalg.norm(ctx_emb)))
            
            # Find unsupported claims
            unsupported = self._find_unsupported_claims(response, context)
            
            # Strict criteria: Reject if similarity too low OR any unsupported claims
            if similarity < self.similarity_threshold:
                logger.warning(f"Poor grounding score: {similarity:.2f}")
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=STANDARD_REFUSAL,
                    reasoning=f"Insufficient grounding (score: {similarity:.2f})",
                    grounding_score=similarity,
                    violation=RegulatoryViolation.INSUFFICIENT_GROUNDING,
                    confidence=0.9
                )
            
            if unsupported:
                logger.warning(f"Unsupported claims found: {len(unsupported)}")
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=STANDARD_REFUSAL,
                    reasoning="Unsupported claims in response",
                    grounding_score=similarity,
                    unsupported_claims=unsupported,
                    violation=RegulatoryViolation.INACCURATE_CLAIMS,
                    confidence=0.9
                )
            
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Grounding check passed",
                grounding_score=similarity,
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Grounding check failed: {e}")
            # Be conservative on error
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=STANDARD_REFUSAL,
                reasoning="Grounding check error",
                violation=RegulatoryViolation.INSUFFICIENT_GROUNDING,
                confidence=0.5
            )
    
    def _find_unsupported_claims(self, response: str, context: str) -> List[str]:
        """Find claims in response not supported by context"""
        unsupported = []
        sentences = re.split(r'[.!?]+', response)
        context_lower = context.lower()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            
            # Skip meta phrases and disclaimers
            skip_phrases = [
                "i don't", "i cannot", "please consult", "medication guide",
                "healthcare provider", "this is not", "see the", "call your doctor"
            ]
            if any(phrase in sentence.lower() for phrase in skip_phrases):
                continue
            
            sentence_lower = sentence.lower()
            
            # Check for specific medical claims not in context
            medical_terms = re.findall(r'\b(?:cause|treat|cure|prevent|interact|contraindicated)\b', sentence_lower)
            for term in medical_terms:
                # Check if the specific claim is in context
                claim_words = sentence_lower.split()
                claim_found = False
                for i, word in enumerate(claim_words):
                    if word == term:
                        # Get surrounding context (3 words before and after)
                        start = max(0, i-3)
                        end = min(len(claim_words), i+4)
                        claim_phrase = ' '.join(claim_words[start:end])
                        if claim_phrase not in context_lower:
                            unsupported.append(sentence)
                            claim_found = True
                            break
                if claim_found:
                    break
            
            # Check for specific numbers/dosages not in context
            numbers = re.findall(r'\b\d+\s*(?:mg|ml|hours?|days?|weeks?|tablet|pill)\b', sentence_lower)
            for num in numbers:
                if num not in context_lower:
                    unsupported.append(sentence)
                    break
        
        return unsupported
    
    def _check_compliance_patterns(self, response: str, query: str) -> ValidationDecision:
        """Check for regulatory compliance violations"""
        response_lower = response.lower()
        
        # Risk communication - auto-add disclaimer for side effects
        if self._is_listing_side_effects(response_lower) and not self._has_disclaimer(response_lower):
            corrected = response.rstrip('.') + ". This is not a complete list. See the Medication Guide for full information."
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=corrected,
                reasoning="Auto-added safety disclaimer",
                violation=RegulatoryViolation.INADEQUATE_RISK_COMM,
                confidence=0.95
            )
        
        # Check for medical advice
        if re.search(r"you should (?:take|stop|increase|decrease)", response_lower):
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=SAFETY_REFUSAL,
                reasoning="Medical advice detected",
                violation=RegulatoryViolation.MEDICAL_ADVICE,
                confidence=0.95
            )
        
        # Check for unsupported reassurances
        if any(phrase in response_lower for phrase in ["don't worry", "perfectly safe", "nothing to worry"]):
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=SAFETY_REFUSAL,
                reasoning="Inappropriate reassurance",
                violation=RegulatoryViolation.INACCURATE_CLAIMS,
                confidence=0.95
            )
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="Compliance check passed",
            confidence=0.95
        )
    
    def _is_listing_side_effects(self, text: str) -> bool:
        """Check if response lists side effects"""
        indicators = ['side effect', 'adverse', 'may cause', 'can cause', 'include']
        return sum(1 for ind in indicators if ind in text) >= 2
    
    def _has_disclaimer(self, text: str) -> bool:
        """Check if response has required disclaimer"""
        disclaimers = ['not a complete list', 'medication guide', 'full information']
        return any(disc in text for disc in disclaimers)

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Single global instance
hybrid_guard = HybridSafetyGuard()

# Legacy compatibility aliases
unified_guard = hybrid_guard
enhanced_guard = hybrid_guard
persona_validator = hybrid_guard
simple_guard = hybrid_guard

# Legacy function for backward compatibility
def evaluate_response(context: str, user_question: str, assistant_response: str, **kwargs) -> Tuple[bool, str, str]:
    """Legacy compatibility function"""
    import asyncio
    
    async def validate():
        return await hybrid_guard.validate_response(
            response=assistant_response,
            context=context,
            query=user_question
        )
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, validate())
                result = future.result()
        else:
            result = loop.run_until_complete(validate())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(validate())
    
    is_safe = (result.result == ValidationResult.APPROVED)
    return is_safe, result.final_response, result.reasoning