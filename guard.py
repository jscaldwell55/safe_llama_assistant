# guard.py - Complete Updated Safety System with All Fixes
"""
Consolidated safety system combining:
1. Document grounding via embedding similarity
2. Pattern-based rules for deterministic blocking
3. All 9 regulatory categories
4. Violence/harm detection
5. Illegal drug detection
6. Strict grounding for interactions
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
    """9 Core Regulatory Categories"""
    INACCURATE_CLAIMS = "inaccurate_claims"           # Cat 1
    INADEQUATE_RISK_COMM = "inadequate_risk_comm"     # Cat 2
    OFF_LABEL_USE = "off_label_use"                   # Cat 3
    IMPROPER_PROMOTION = "improper_promotion"         # Cat 4
    CROSS_PRODUCT_REF = "cross_product_ref"           # Cat 5
    MEDICAL_ADVICE = "medical_advice"                 # Cat 6
    SAFETY_CRITICAL_MISS = "safety_critical_miss"     # Cat 7
    ADMIN_MISUSE = "admin_misuse"                     # Cat 8
    UNAPPROVED_DOSING = "unapproved_dosing"          # Cat 9
    OFF_TOPIC = "off_topic"                          # Non-medical queries
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
    llm_safety_score: float = 0.0
    unsupported_claims: List[str] = field(default_factory=list)
    violation: RegulatoryViolation = RegulatoryViolation.NONE
    should_log: bool = False

# ============================================================================
# PRE-APPROVED RESPONSE TEMPLATES
# ============================================================================

SAFETY_REFUSAL = "I'm sorry, I cannot discuss that. Would you like to talk about something else?"
NO_INFO_REFUSAL = "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"

REFUSAL_TEMPLATES = {
    RegulatoryViolation.MEDICAL_ADVICE: SAFETY_REFUSAL,
    RegulatoryViolation.UNAPPROVED_DOSING: SAFETY_REFUSAL,
    RegulatoryViolation.OFF_LABEL_USE: SAFETY_REFUSAL,
    RegulatoryViolation.ADMIN_MISUSE: SAFETY_REFUSAL,
    RegulatoryViolation.IMPROPER_PROMOTION: SAFETY_REFUSAL,
    RegulatoryViolation.CROSS_PRODUCT_REF: SAFETY_REFUSAL,
    RegulatoryViolation.SAFETY_CRITICAL_MISS: SAFETY_REFUSAL,
    RegulatoryViolation.INADEQUATE_RISK_COMM: SAFETY_REFUSAL,
    RegulatoryViolation.INACCURATE_CLAIMS: NO_INFO_REFUSAL,
    RegulatoryViolation.OFF_TOPIC: NO_INFO_REFUSAL,
    "default": NO_INFO_REFUSAL
}

# ============================================================================
# HYBRID SAFETY GUARD
# ============================================================================

class HybridSafetyGuard:
    """
    Unified safety system using:
    1. Embedding similarity for grounding
    2. Pattern rules for deterministic blocking
    3. Violence/harm detection
    4. Illegal substance detection
    """
    
    def __init__(self):
        self.enabled = True
        self.similarity_threshold = 0.35
        self.llm_safety_threshold = 0.7
        self.embedding_model = None
        self.use_llm_guard = False  # Default to False
        
        # Load configuration
        try:
            from config import ENABLE_GUARD, SEMANTIC_SIMILARITY_THRESHOLD, USE_LLM_GUARD, LLM_CONFIDENCE_THRESHOLD
            self.enabled = ENABLE_GUARD
            self.similarity_threshold = SEMANTIC_SIMILARITY_THRESHOLD
            self.use_llm_guard = USE_LLM_GUARD
            self.llm_safety_threshold = LLM_CONFIDENCE_THRESHOLD
            logger.info(f"Guard config loaded - Enabled: {self.enabled}, LLM guard: {self.use_llm_guard}")
        except ImportError:
            logger.warning("Using default configuration - LLM guard disabled")
            self.use_llm_guard = False
        
        self._load_models()
    
    def _load_models(self):
        """Load embedding model and confirm LLM availability"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Embedding model loaded for grounding")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
        
        if self.use_llm_guard:
            logger.info("LLM guard enabled for safety assessment")
    
    # ========================================================================
    # QUERY VALIDATION - ENHANCED WITH VIOLENCE/HARM DETECTION
    # ========================================================================
    
    async def validate_query(self, query: str) -> Optional[ValidationDecision]:
        """Pre-screen queries using patterns"""
        if not self.enabled:
            return None
        
        query_lower = query.lower()
        
        # CRITICAL: Check for violence/harm/dangerous intent FIRST
        violence_patterns = [
            r'kill\s+(?:someone|myself|anybody|anyone|a person)',
            r'(?:lethal|deadly|fatal)\s+(?:dose|amount)',
            r'(?:overdose|od)\s+on',
            r'(?:suicide|harm myself|hurt myself)',
            r'(?:poison|hurt|harm)\s+(?:someone|anybody|anyone)',
            r'how much.*(?:kill|die|lethal|fatal)',
            r'(?:end my life|take my life)',
        ]
        
        for pattern in violence_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"CRITICAL: Query blocked for violence/harm intent: {query[:50]}...")
                return self._create_redirect(
                    RegulatoryViolation.ADMIN_MISUSE, 
                    "Query about using medication to cause harm"
                )
        
        # Check for illegal drug use
        illegal_drugs = ['cocaine', 'heroin', 'meth', 'crack', 'ecstasy', 'mdma', 'lsd', 'mushrooms', 'pcp', 'weed', 'marijuana']
        for drug in illegal_drugs:
            if drug in query_lower:
                logger.warning(f"Query blocked: illegal drug use ({drug})")
                return self._create_redirect(RegulatoryViolation.ADMIN_MISUSE, f"Query about illegal drug use: {drug}")
        
        # Off-topic detection
        if self._is_clearly_off_topic(query_lower):
            return self._create_redirect(RegulatoryViolation.OFF_TOPIC, "Off-topic request")
        
        # Check for clear unsafe patterns
        unsafe_patterns = [
            # Dosing modifications
            (r"(?:how much|what dose).*(?:should i take|do i need)", 
             RegulatoryViolation.UNAPPROVED_DOSING, "personal dosing inquiry"),
            (r"(?:can i |should i |is it safe to )(?:double|triple|increase|decrease|skip).*(?:dose|medication)", 
             RegulatoryViolation.UNAPPROVED_DOSING, "dose modification request"),
            
            # Pediatric/Off-label
            (r"(?:can i give|should i give|is it safe to give).*(?:to my|to a).*(?:child|kid|baby|infant)", 
             RegulatoryViolation.OFF_LABEL_USE, "pediatric administration request"),
            (r"(?:let|have|give).*(?:my |a )(?:son|daughter|child|kid|baby).*(?:take|have)", 
             RegulatoryViolation.OFF_LABEL_USE, "pediatric use request"),
            
            # Clear misuse
            (r"(?:can i |should i |how to )(?:snort|inject|smoke|crush).*(?:journvax|medication|it)", 
             RegulatoryViolation.ADMIN_MISUSE, "unsafe administration method"),
            (r"(?:can i |should i )share.*(?:prescription|medication)", 
             RegulatoryViolation.ADMIN_MISUSE, "prescription sharing request"),
            
            # Alcohol/substance mixing
            (r"(?:while|when|after).*(?:drinking|drunk|alcohol|beer|wine|liquor)", 
             RegulatoryViolation.ADMIN_MISUSE, "alcohol interaction query"),
        ]
        
        for pattern, violation, reason in unsafe_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Query pattern match: {reason}")
                return self._create_redirect(violation, reason)
        
        # LLM assessment only for ambiguous cases
        if self.use_llm_guard and self._needs_llm_query_check(query_lower):
            llm_decision = await self._llm_assess_query(query)
            if llm_decision:
                return llm_decision
        
        return None
    
    def _is_clearly_off_topic(self, query_lower: str) -> bool:
        """Check if query is clearly off-topic"""
        journvax_keywords = ['journvax', 'medication', 'side effect', 'adverse', 'dose', 'prescription', 'drug']
        off_topic_keywords = ['bedtime story', 'poem', 'code', 'python', 'javascript', 'weather', 'recipe', 'math']
        
        has_journvax = any(kw in query_lower for kw in journvax_keywords)
        has_off_topic = any(kw in query_lower for kw in off_topic_keywords)
        
        return has_off_topic and not has_journvax
    
    def _needs_llm_query_check(self, query_lower: str) -> bool:
        """Check if query needs LLM assessment"""
        personal_indicators = ['my ', 'i ', "i'm ", 'should i', 'can i', 'do i', 'am i']
        medical_terms = ['pregnant', 'breastfeed', 'allerg', 'condition', 'disease', 'medication']
        
        has_personal = any(ind in query_lower for ind in personal_indicators)
        has_medical = any(term in query_lower for term in medical_terms)
        
        return has_personal and has_medical
    
    async def _llm_assess_query(self, query: str) -> Optional[ValidationDecision]:
        """Use LLM to assess query safety - currently disabled"""
        # LLM guard is disabled to prevent false positives
        return None
    
    # ========================================================================
    # RESPONSE VALIDATION - ENHANCED GROUNDING CHECK
    # ========================================================================
    
    async def validate_response(
        self,
        response: str,
        context: str = "",
        query: str = "",
        **kwargs
    ) -> ValidationDecision:
        """Validate response using multiple methods"""
        if not self.enabled:
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation disabled",
                confidence=1.0
            )
        
        # Skip validation for refusals
        if response.lower().startswith(("i cannot", "i don't have", "i'm not able", "i'm sorry")):
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Refusal response - no validation needed",
                confidence=0.95
            )
        
        # STEP 1: Pattern-based compliance check
        compliance_result = self._check_compliance_patterns(response, query)
        if compliance_result.result != ValidationResult.APPROVED:
            return compliance_result
        
        # STEP 2: Grounding check (STRICTER)
        if context and len(context) > 50 and self.embedding_model:
            grounding_result = self._check_grounding(response, context)
            compliance_result.grounding_score = grounding_result.grounding_score
            
            if grounding_result.result != ValidationResult.APPROVED:
                return grounding_result
        
        # STEP 3: LLM safety assessment (currently disabled)
        if self.use_llm_guard:
            llm_result = await self._llm_assess_response(response, context, query)
            if llm_result and llm_result.result != ValidationResult.APPROVED:
                return llm_result
            if llm_result:
                compliance_result.llm_safety_score = llm_result.confidence
        else:
            logger.debug("LLM safety assessment skipped (disabled in config)")
        
        return compliance_result
    
    def _check_compliance_patterns(self, response: str, query: str) -> ValidationDecision:
        """Check response for compliance violations"""
        response_lower = response.lower()
        
        # Auto-corrections for specific categories
        
        # Category 2: Risk Communication - Add disclaimer if listing side effects
        if self._is_listing_side_effects(response_lower) and not self._has_disclaimer(response_lower):
            corrected = response.rstrip('.') + ". This is not a complete list. See the Medication Guide for full information."
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=corrected,
                reasoning="Auto-added medical disclaimer",
                violation=RegulatoryViolation.INADEQUATE_RISK_COMM,
                confidence=0.95
            )
        
        # Category 7: Safety-Critical - Add emergency guidance if needed
        if self._mentions_severe_symptoms(response_lower) and not self._has_emergency_guidance(response_lower):
            corrected = response.rstrip('.') + ". If you experience severe symptoms, seek immediate medical attention."
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=corrected,
                reasoning="Auto-added emergency guidance",
                violation=RegulatoryViolation.SAFETY_CRITICAL_MISS,
                confidence=0.95
            )
        
        # Check for other violations that can't be auto-corrected
        
        # Category 1: Inaccurate Claims
        if re.search(r"(?:doesn't|don't) mention.*(?:so|therefore).*(?:fine|safe|ok)", response_lower):
            return self._create_rejection(RegulatoryViolation.INACCURATE_CLAIMS, "Implied safety from absence")
        
        # Category 4: Improper tone
        problematic_phrases = ["don't worry", "should be fine", "perfectly safe", "nothing to worry about"]
        for phrase in problematic_phrases:
            if phrase in response_lower:
                return self._create_rejection(RegulatoryViolation.IMPROPER_PROMOTION, f"Inappropriate reassurance: {phrase}")
        
        # Category 6: Medical Advice
        medical_advice_patterns = [
            r"you should (?:take|stop|increase|decrease) your",
            r"(?:double|halve|skip) your (?:dose|medication)",
            r"i (?:recommend|suggest|advise) you"
        ]
        
        for pattern in medical_advice_patterns:
            if re.search(pattern, response_lower):
                return self._create_rejection(RegulatoryViolation.MEDICAL_ADVICE, "Personal medical advice")
        
        # Category 9: Unapproved dosing
        if re.search(r"take \d+\s*(?:tablet|pill|mg|ml)", response_lower):
            if not re.search(r'(?:according to|per|as stated in|medication guide)', response_lower):
                return self._create_rejection(RegulatoryViolation.UNAPPROVED_DOSING, "Specific dosing without citation")
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="Pattern compliance check passed",
            confidence=0.95
        )
    
    def _check_grounding(self, response: str, context: str) -> ValidationDecision:
        """Check if response is grounded in context - STRICTER for safety"""
        try:
            resp_emb = self.embedding_model.encode(response, show_progress_bar=False)
            ctx_emb = self.embedding_model.encode(context, show_progress_bar=False)
            similarity = float(np.dot(resp_emb, ctx_emb) / (np.linalg.norm(resp_emb) * np.linalg.norm(ctx_emb)))
            
            unsupported = self._find_unsupported_claims(response, context)
            
            # STRICTER: Fail if ANY unsupported interaction claims (like grapefruit)
            # OR low similarity with multiple unsupported claims
            if unsupported:
                # Check if any unsupported claim is about interactions/substances
                for claim in unsupported:
                    claim_lower = claim.lower()
                    if any(word in claim_lower for word in ['grapefruit', 'avoid', 'interact', 'alcohol', 'food', 'drink']):
                        logger.warning(f"Critical unsupported interaction claim: {claim}")
                        return ValidationDecision(
                            result=ValidationResult.REJECTED,
                            final_response=NO_INFO_REFUSAL,
                            reasoning=f"Unsupported interaction/substance claim",
                            grounding_score=similarity,
                            unsupported_claims=unsupported,
                            violation=RegulatoryViolation.INACCURATE_CLAIMS
                        )
            
            # Also fail if low similarity AND any unsupported claims
            if similarity < self.similarity_threshold and len(unsupported) > 0:
                logger.warning(f"Poor grounding: {similarity:.2f}, unsupported: {len(unsupported)}")
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=NO_INFO_REFUSAL,
                    reasoning=f"Poor grounding (score: {similarity:.2f})",
                    grounding_score=similarity,
                    unsupported_claims=unsupported,
                    violation=RegulatoryViolation.INACCURATE_CLAIMS
                )
            
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Grounding check passed",
                grounding_score=similarity
            )
            
        except Exception as e:
            logger.error(f"Grounding check failed: {e}")
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Grounding check error",
                grounding_score=0.5
            )
    
    def _find_unsupported_claims(self, response: str, context: str) -> List[str]:
        """Find claims not supported by context - ENHANCED to catch specific substances"""
        unsupported = []
        sentences = re.split(r'[.!?]+', response)
        context_lower = context.lower()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Skip common safe phrases
            skip_phrases = [
                "i don't", "i cannot", "please consult", "according to",
                "medication guide", "healthcare provider", "this is not"
            ]
            if any(phrase in sentence.lower() for phrase in skip_phrases):
                continue
            
            sentence_lower = sentence.lower()
            
            # CRITICAL: Check for specific food/drug interactions not in context
            # This includes grapefruit and any other specific substances
            interaction_patterns = [
                r'avoid\s+(?:food|drink|beverages?)?\s*(?:containing|with)?\s*(\w+)',
                r'(?:don\'t|do not)\s+(?:take|mix|combine)\s+(?:with|and)\s+(\w+)',
                r'(?:interact|interfere)s?\s+with\s+(\w+)',
            ]
            
            for pattern in interaction_patterns:
                matches = re.findall(pattern, sentence_lower)
                for match in matches:
                    # Check if this specific item is mentioned in context
                    item = match if isinstance(match, str) else str(match)
                    # Exclude generic terms
                    if item not in ['water', 'food', 'it', 'medication', 'this', 'that', 'journvax']:
                        # If specific item not in context, flag it
                        if item not in context_lower:
                            unsupported.append(sentence)
                            logger.warning(f"Unsupported interaction claim: {item} not found in context")
                            break
            
            # Check for specific substances mentioned that aren't in context
            specific_substances = ['grapefruit', 'alcohol', 'caffeine', 'tobacco', 'marijuana', 'cannabis']
            for substance in specific_substances:
                if substance in sentence_lower and substance not in context_lower:
                    if sentence not in unsupported:  # Avoid duplicates
                        unsupported.append(sentence)
                        logger.warning(f"Unsupported substance claim: {substance} not in context")
                    break
            
            # Check for "avoid X" or "take with Y" patterns where X/Y isn't mentioned
            specific_items = re.findall(r'\b(?:avoid|take with|do not take with|limit)\s+(\w+)', sentence_lower)
            for item in specific_items:
                # Only flag if it's a specific substance, not generic terms
                if item not in context_lower and item not in ['water', 'food', 'it', 'medication', 'this', 'that']:
                    if sentence not in unsupported:  # Avoid duplicates
                        unsupported.append(sentence)
                    break
            
            # Check for specific numbers not in context
            numbers = re.findall(r'\b\d+\s*(?:mg|ml|percent|%)\b', sentence_lower)
            for num in numbers:
                if num not in context_lower:
                    if sentence not in unsupported:  # Avoid duplicates
                        unsupported.append(sentence)
                    break
        
        return unsupported
    
    async def _llm_assess_response(self, response: str, context: str, query: str) -> Optional[ValidationDecision]:
        """Use LLM for response safety assessment - currently disabled"""
        # LLM guard is disabled to prevent false positives
        if True:  # Always disabled for now
            return None
        
        # Original LLM assessment code (kept for reference but not executed)
        return None
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _is_listing_side_effects(self, text: str) -> bool:
        """Check if response is listing side effects"""
        indicators = ['side effect', 'adverse', 'may cause', 'can cause', 'reported', 'common', 'include']
        return sum(1 for ind in indicators if ind in text) >= 2
    
    def _has_disclaimer(self, text: str) -> bool:
        """Check if response has required disclaimer"""
        disclaimers = ['not a complete list', 'medication guide', 'full information', 'complete safety']
        return any(disc in text for disc in disclaimers)
    
    def _mentions_severe_symptoms(self, text: str) -> bool:
        """Check if response mentions severe symptoms"""
        severe = ['severe', 'emergency', 'trouble breathing', 'chest pain', 'swelling', 'anaphylaxis', 'allergic reaction']
        return any(s in text for s in severe)
    
    def _has_emergency_guidance(self, text: str) -> bool:
        """Check if response has emergency guidance"""
        emergency = ['seek immediate', 'call 911', 'emergency', 'medical attention', 'immediately']
        return any(e in text for e in emergency)
    
    def _create_redirect(self, violation: RegulatoryViolation, reason: str) -> ValidationDecision:
        """Create redirect for unsafe queries"""
        return ValidationDecision(
            result=ValidationResult.REDIRECT,
            final_response=REFUSAL_TEMPLATES.get(violation, REFUSAL_TEMPLATES["default"]),
            reasoning=reason,
            confidence=0.95,
            violation=violation,
            should_log=True
        )
    
    def _create_rejection(self, violation: RegulatoryViolation, reason: str) -> ValidationDecision:
        """Create rejection for non-compliant responses"""
        return ValidationDecision(
            result=ValidationResult.REJECTED,
            final_response=REFUSAL_TEMPLATES.get(violation, REFUSAL_TEMPLATES["default"]),
            reasoning=reason,
            confidence=0.90,
            violation=violation,
            should_log=True
        )
    
    def get_validation_summary(self, decision: ValidationDecision) -> Dict:
        """Get summary of validation decision"""
        return {
            "approved": decision.result == ValidationResult.APPROVED,
            "violation": decision.violation.value,
            "grounding_score": decision.grounding_score,
            "llm_safety_score": decision.llm_safety_score,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning
        }

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Single global instance
hybrid_guard = HybridSafetyGuard()

# Legacy compatibility
unified_guard = hybrid_guard
enhanced_guard = hybrid_guard
persona_validator = hybrid_guard
simple_guard = hybrid_guard

# Legacy function
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