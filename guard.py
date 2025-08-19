# guard.py - Unified Safety System with Hybrid Validation
"""
Consolidated safety system combining:
1. Document grounding via embedding similarity
2. LLM-based safety assessment for nuanced violations
3. Pattern-based rules for deterministic blocking
4. All 9 regulatory categories

Uses both embedding model AND Llama for intelligent validation.
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
    llm_safety_score: float = 0.0  # New: LLM assessment score
    unsupported_claims: List[str] = field(default_factory=list)
    violation: RegulatoryViolation = RegulatoryViolation.NONE
    should_log: bool = False

# ============================================================================
# PRE-APPROVED RESPONSE TEMPLATES (SIMPLIFIED)
# ============================================================================

# Two standard responses for all redirects
SAFETY_REFUSAL = "I'm sorry, I cannot discuss that. Would you like to talk about something else?"
NO_INFO_REFUSAL = "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"

# Map violations to response types
REFUSAL_TEMPLATES = {
    # Safety violations use the safety refusal
    RegulatoryViolation.MEDICAL_ADVICE: SAFETY_REFUSAL,
    RegulatoryViolation.UNAPPROVED_DOSING: SAFETY_REFUSAL,
    RegulatoryViolation.OFF_LABEL_USE: SAFETY_REFUSAL,
    RegulatoryViolation.ADMIN_MISUSE: SAFETY_REFUSAL,
    RegulatoryViolation.IMPROPER_PROMOTION: SAFETY_REFUSAL,
    RegulatoryViolation.CROSS_PRODUCT_REF: SAFETY_REFUSAL,
    RegulatoryViolation.SAFETY_CRITICAL_MISS: SAFETY_REFUSAL,
    RegulatoryViolation.INADEQUATE_RISK_COMM: SAFETY_REFUSAL,
    
    # No information uses the no-info refusal
    RegulatoryViolation.INACCURATE_CLAIMS: NO_INFO_REFUSAL,
    RegulatoryViolation.OFF_TOPIC: NO_INFO_REFUSAL,
    
    # Default fallback
    "default": NO_INFO_REFUSAL
}

# ============================================================================
# HYBRID SAFETY GUARD (EMBEDDINGS + LLM)
# ============================================================================

class HybridSafetyGuard:
    """
    Unified safety system using BOTH:
    1. Embedding similarity for grounding
    2. Llama model for nuanced safety assessment
    3. Pattern rules for deterministic blocking
    """
    
    def __init__(self):
        self.enabled = True
        self.similarity_threshold = 0.35
        self.llm_safety_threshold = 0.7
        self.embedding_model = None
        self.use_llm_guard = False
        
        # Load configuration
        try:
            from config import ENABLE_GUARD, SEMANTIC_SIMILARITY_THRESHOLD, USE_LLM_GUARD, LLM_CONFIDENCE_THRESHOLD
            self.enabled = ENABLE_GUARD
            self.similarity_threshold = SEMANTIC_SIMILARITY_THRESHOLD
            self.use_llm_guard = USE_LLM_GUARD
            self.llm_safety_threshold = LLM_CONFIDENCE_THRESHOLD
        except ImportError:
            logger.warning("Using default configuration")
        
        self._load_models()
    
    def _load_models(self):
        """Load embedding model and confirm LLM availability"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Embedding model loaded for grounding")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
        
        # Confirm LLM is available if enabled
        if self.use_llm_guard:
            logger.info("LLM guard enabled for safety assessment")
    
    # ========================================================================
    # QUERY VALIDATION (Pattern-based + Optional LLM)
    # ========================================================================
    
    async def validate_query(self, query: str) -> Optional[ValidationDecision]:
        """
        Pre-screen queries using patterns and optionally LLM.
        Step 1: Deterministic pattern matching (fast)
        Step 2: LLM assessment for ambiguous cases (if enabled)
        """
        if not self.enabled:
            return None
        
        query_lower = query.lower()
        
        # STEP 1: DETERMINISTIC PATTERN CHECKS (Fast, reliable)
        
        # Off-topic detection
        if self._is_clearly_off_topic(query_lower):
            return self._create_redirect(RegulatoryViolation.OFF_TOPIC, "Off-topic request")
        
        # Clear unsafe patterns
        unsafe_patterns = [
            # Dosing
            (r"(?:how much|how many|what dose).*(?:should|can|do).*(?:i|we|they)", 
             RegulatoryViolation.UNAPPROVED_DOSING, "dosing inquiry"),
            (r"(?:double|triple|increase|extra).*(?:dose|medication)", 
             RegulatoryViolation.UNAPPROVED_DOSING, "dose modification"),
            # Pediatric/Off-label
            (r"(?:give|can i give).*(?:to my|to a).*(?:child|kid|baby)", 
             RegulatoryViolation.OFF_LABEL_USE, "pediatric use"),
            # Misuse
            (r"(?:snort|inject|smoke|crush).*(?:journvax|medication|it)", 
             RegulatoryViolation.ADMIN_MISUSE, "unsafe administration"),
            (r"share.*(?:prescription|medication)", 
             RegulatoryViolation.ADMIN_MISUSE, "prescription sharing"),
        ]
        
        for pattern, violation, reason in unsafe_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Pattern match: {reason}")
                return self._create_redirect(violation, reason)
        
        # STEP 2: LLM ASSESSMENT (If enabled and query is ambiguous)
        if self.use_llm_guard and self._is_potentially_unsafe(query_lower):
            llm_decision = await self._llm_assess_query(query)
            if llm_decision:
                return llm_decision
        
        return None  # Query is safe
    
    def _is_clearly_off_topic(self, query_lower: str) -> bool:
        """Deterministic off-topic check"""
        journvax_keywords = ['journvax', 'medication', 'side effect', 'dose', 'prescription', 'drug']
        off_topic_keywords = ['bedtime story', 'poem', 'code', 'python', 'javascript', 'weather', 'recipe']
        
        has_journvax = any(kw in query_lower for kw in journvax_keywords)
        has_off_topic = any(kw in query_lower for kw in off_topic_keywords)
        
        return has_off_topic and not has_journvax
    
    def _is_potentially_unsafe(self, query_lower: str) -> bool:
        """Check if query might need LLM assessment"""
        ambiguous_terms = ['safe', 'okay', 'can i', 'should i', 'what if', 'is it bad']
        return any(term in query_lower for term in ambiguous_terms)
    
    async def _llm_assess_query(self, query: str) -> Optional[ValidationDecision]:
        """Use LLM to assess query safety"""
        try:
            from llm_client import call_huggingface_with_retry
            from config import GUARD_MODEL_PARAMS
            
            prompt = f"""Assess if this pharmaceutical query is safe to answer.

Query: "{query}"

Respond with JSON:
{{
  "is_safe": true/false,
  "violation_type": "none|medical_advice|dosing|off_label|misuse",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Focus on these violations:
- medical_advice: Individual treatment decisions
- dosing: Specific dosage recommendations  
- off_label: Unapproved uses (pediatric, pregnancy)
- misuse: Unsafe administration, sharing

JSON:"""
            
            response = await call_huggingface_with_retry(prompt, GUARD_MODEL_PARAMS)
            
            # Parse LLM response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    if not result.get('is_safe', True) and result.get('confidence', 0) > self.llm_safety_threshold:
                        violation_map = {
                            'medical_advice': RegulatoryViolation.MEDICAL_ADVICE,
                            'dosing': RegulatoryViolation.UNAPPROVED_DOSING,
                            'off_label': RegulatoryViolation.OFF_LABEL_USE,
                            'misuse': RegulatoryViolation.ADMIN_MISUSE
                        }
                        
                        violation = violation_map.get(
                            result.get('violation_type', 'none'),
                            RegulatoryViolation.MEDICAL_ADVICE
                        )
                        
                        logger.info(f"LLM flagged query: {result.get('reasoning', 'No reason')}")
                        
                        return self._create_redirect(
                            violation,
                            f"LLM assessment: {result.get('reasoning', 'Unsafe query')}"
                        )
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response as JSON")
                
        except Exception as e:
            logger.error(f"LLM query assessment failed: {e}")
        
        return None
    
    # ========================================================================
    # RESPONSE VALIDATION (Embeddings + Patterns + Optional LLM)
    # ========================================================================
    
    async def validate_response(
        self,
        response: str,
        context: str = "",
        query: str = "",
        **kwargs
    ) -> ValidationDecision:
        """
        Validate response using multiple methods:
        1. Pattern-based compliance checking (deterministic)
        2. Embedding similarity for grounding (if context provided)
        3. LLM safety assessment (if enabled)
        """
        if not self.enabled:
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation disabled",
                confidence=1.0
            )
        
        # Skip validation for clear refusals
        if response.lower().startswith(("i cannot", "i don't have", "i'm not able")):
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Refusal response - no validation needed",
                confidence=0.95
            )
        
        # STEP 1: PATTERN-BASED COMPLIANCE CHECK
        compliance_result = self._check_compliance_patterns(response, query)
        if compliance_result.result != ValidationResult.APPROVED:
            return compliance_result
        
        # STEP 2: GROUNDING CHECK (Embedding similarity)
        if context and len(context) > 50 and self.embedding_model:
            grounding_result = self._check_grounding(response, context)
            compliance_result.grounding_score = grounding_result.grounding_score
            
            if grounding_result.result != ValidationResult.APPROVED:
                return grounding_result
        
        # STEP 3: LLM SAFETY ASSESSMENT (If enabled)
        if self.use_llm_guard:
            llm_result = await self._llm_assess_response(response, context, query)
            if llm_result and llm_result.result != ValidationResult.APPROVED:
                return llm_result
            if llm_result:
                compliance_result.llm_safety_score = llm_result.confidence
        
        return compliance_result
    
    def _check_compliance_patterns(self, response: str, query: str) -> ValidationDecision:
        """Deterministic pattern-based compliance check"""
        response_lower = response.lower()
        
        # Category 1: Inaccurate Claims
        if re.search(r"(?:doesn't|don't) mention.*(?:so|therefore).*(?:fine|safe|ok)", response_lower):
            return self._create_rejection(RegulatoryViolation.INACCURATE_CLAIMS, "Implied safety from absence")
        
        # Category 2: Risk Communication
        if self._has_medical_content(response_lower) and not self._has_disclaimer(response_lower):
            if re.search(r'(include|such as|may cause|reported)', response_lower):
                corrected = response.rstrip('.') + ". This is not a complete list. See the Medication Guide for full information."
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=corrected,
                    reasoning="Missing medical disclaimer",
                    violation=RegulatoryViolation.INADEQUATE_RISK_COMM
                )
        
        # Categories 3-9: Core violations
        violation_patterns = [
            (r"don't worry|should be fine|typically safe", RegulatoryViolation.IMPROPER_PROMOTION),
            (r"you (?:should|must) (?:take|stop|increase)", RegulatoryViolation.MEDICAL_ADVICE),
            (r"take \d+\s*(?:mg|tablet|pill)", RegulatoryViolation.UNAPPROVED_DOSING),
            (r"(?:better|worse) than (?:other|another)", RegulatoryViolation.CROSS_PRODUCT_REF),
            (r"give.*to.*(?:child|kid|baby)", RegulatoryViolation.OFF_LABEL_USE),
        ]
        
        for pattern, violation in violation_patterns:
            if re.search(pattern, response_lower):
                return self._create_rejection(violation, f"Pattern: {pattern[:30]}...")
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="Pattern compliance check passed",
            confidence=0.95
        )
    
    def _check_grounding(self, response: str, context: str) -> ValidationDecision:
        """Check document grounding using embeddings"""
        try:
            # Calculate overall similarity
            resp_emb = self.embedding_model.encode(response, show_progress_bar=False)
            ctx_emb = self.embedding_model.encode(context, show_progress_bar=False)
            similarity = float(np.dot(resp_emb, ctx_emb) / (np.linalg.norm(resp_emb) * np.linalg.norm(ctx_emb)))
            
            # Find unsupported claims
            unsupported = self._find_unsupported_claims(response, context)
            
            # Fail only if similarity is low AND multiple unsupported claims
            if similarity < self.similarity_threshold and len(unsupported) > 2:
                logger.warning(f"Poor grounding: {similarity:.2f}, unsupported: {len(unsupported)}")
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=REFUSAL_TEMPLATES["default"],
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
    
    async def _llm_assess_response(self, response: str, context: str, query: str) -> Optional[ValidationDecision]:
        """Use LLM for nuanced safety assessment"""
        try:
            from llm_client import call_huggingface_with_retry
            from config import GUARD_MODEL_PARAMS
            
            prompt = f"""Assess this pharmaceutical response for safety violations.

User Query: "{query}"

Response: "{response[:500]}"

Context Available: {"Yes" if context else "No"}

Check for these violations:
1. Providing specific medical advice (treatment decisions)
2. Recommending dosage changes or specific doses
3. Minimizing risks ("don't worry", "should be fine")
4. Making claims not supported by provided context
5. Suggesting off-label use (pediatric, pregnancy)

Respond with JSON:
{{
  "is_safe": true/false,
  "violation": "none|medical_advice|dosing|improper_tone|ungrounded|off_label",
  "confidence": 0.0-1.0,
  "specific_issue": "what exactly is problematic",
  "suggestion": "how to fix it"
}}

JSON:"""
            
            llm_response = await call_huggingface_with_retry(prompt, GUARD_MODEL_PARAMS)
            
            # Parse response
            try:
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    if not result.get('is_safe', True) and result.get('confidence', 0) > self.llm_safety_threshold:
                        
                        violation_map = {
                            'medical_advice': RegulatoryViolation.MEDICAL_ADVICE,
                            'dosing': RegulatoryViolation.UNAPPROVED_DOSING,
                            'improper_tone': RegulatoryViolation.IMPROPER_PROMOTION,
                            'ungrounded': RegulatoryViolation.INACCURATE_CLAIMS,
                            'off_label': RegulatoryViolation.OFF_LABEL_USE
                        }
                        
                        violation = violation_map.get(
                            result.get('violation', 'none'),
                            RegulatoryViolation.MEDICAL_ADVICE
                        )
                        
                        logger.info(f"LLM flagged response: {result.get('specific_issue', 'No details')}")
                        
                        # Use LLM's suggestion if available, otherwise use template
                        final_response = result.get('suggestion', REFUSAL_TEMPLATES.get(violation, REFUSAL_TEMPLATES["default"]))
                        
                        return ValidationDecision(
                            result=ValidationResult.REJECTED,
                            final_response=final_response,
                            reasoning=f"LLM: {result.get('specific_issue', 'Safety violation')}",
                            confidence=result.get('confidence', 0.8),
                            llm_safety_score=result.get('confidence', 0.8),
                            violation=violation
                        )
                        
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM safety assessment")
                
        except Exception as e:
            logger.error(f"LLM response assessment failed: {e}")
        
        return None
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
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
    
    def _has_medical_content(self, text: str) -> bool:
        """Check if text contains medical information"""
        medical_terms = ['side effect', 'adverse', 'reaction', 'symptom', 'warning']
        return any(term in text for term in medical_terms)
    
    def _has_disclaimer(self, text: str) -> bool:
        """Check for required disclaimers"""
        disclaimers = ['not a complete list', 'medication guide', 'consult your healthcare']
        return any(disc in text for disc in disclaimers)
    
    def _find_unsupported_claims(self, response: str, context: str) -> List[str]:
        """Find claims not supported by context"""
        unsupported = []
        sentences = re.split(r'[.!?]+', response)
        context_lower = context.lower()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Skip meta-statements
            if any(phrase in sentence.lower() for phrase in ["i don't", "i cannot", "please consult", "according to"]):
                continue
            
            # Check for specific items not in context
            items = re.findall(r'\b(avoid|take with|limit)\s+(\w+)', sentence.lower())
            for _, item in items:
                if item not in context_lower and item not in ['water', 'food', 'it', 'medication']:
                    unsupported.append(sentence)
                    break
        
        return unsupported
    
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