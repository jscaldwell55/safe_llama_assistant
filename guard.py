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
    UNAPPROVED_DOSING = "unapproved_dosing"           # Cat 9
    OFF_TOPIC = "off_topic"                           # Non-medical queries
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

SAFETY_REFUSAL = "I'm sorry, I cannot discuss that. Would you like to talk about something else?"
NO_INFO_REFUSAL = "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"

REFUSAL_TEMPLATES = {
    RegulatoryViolation.MEDICAL_ADVICE: SAFETY_REFUSAL,
    RegulatoryViolation.UNAPPROVED_DOSING: SAFETY_REFUSAL,
    RegulatoryViolation.OFF_LABEL_USE: SAFETY_REFUSAL,
    RegulatoryViolation.ADMIN_MISUSE: SAFETY_REFUSAL,
    RegulatoryViolation.IMPROPER_PROMOTION: SAFETY_REFUSAL,
    RegulatoryViolation.CROSS_PRODUCT_REF: SAFETY_REFUSAL,
    RegulatoryViolation.SAFETY_CRITICAL_MISS: SAFETY_REFUSAL,       # only used if we can't auto-correct
    RegulatoryViolation.INADEQUATE_RISK_COMM: NO_INFO_REFUSAL,      # only used if we can't auto-correct
    RegulatoryViolation.INACCURATE_CLAIMS: NO_INFO_REFUSAL,
    RegulatoryViolation.OFF_TOPIC: NO_INFO_REFUSAL,
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
        self.llm_safety_threshold = 0.85
        self.embedding_model = None
        self.use_llm_guard = False
        
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
        
        if self.use_llm_guard:
            logger.info("LLM guard enabled for safety assessment")
    
    # ========================================================================
    # QUERY VALIDATION (Pattern-based + Optional LLM)
    # ========================================================================
    
    async def validate_query(self, query: str) -> Optional[ValidationDecision]:
        if not self.enabled:
            return None
        
        ql = query.lower()
        
        # Off-topic detection
        if self._is_clearly_off_topic(ql):
            return self._create_redirect(RegulatoryViolation.OFF_TOPIC, "Off-topic request")
        
        # Deterministic unsafe patterns
        unsafe = [
            (r"(?:how much|how many|what dose).*(?:should|can|do).*(?:i|we|they)", RegulatoryViolation.UNAPPROVED_DOSING, "dosing inquiry"),
            (r"(?:double|triple|increase|extra).*(?:dose|medication)", RegulatoryViolation.UNAPPROVED_DOSING, "dose modification"),
            (r"(?:give|can i give).*(?:to my|to a).*(?:child|kid|baby)", RegulatoryViolation.OFF_LABEL_USE, "pediatric use"),
            (r"(?:snort|inject|smoke|crush).*(?:journvax|medication|it)", RegulatoryViolation.ADMIN_MISUSE, "unsafe administration"),
            (r"share.*(?:prescription|medication)", RegulatoryViolation.ADMIN_MISUSE, "prescription sharing"),
        ]
        for pat, viol, why in unsafe:
            if re.search(pat, ql):
                logger.info(f"Pattern match: {why}")
                return self._create_redirect(viol, why)
        
        # Optional LLM check for ambiguous risk-seeking language
        if self.use_llm_guard and self._is_potentially_unsafe(ql):
            llm_decision = await self._llm_assess_query(query)
            if llm_decision:
                return llm_decision
        
        return None
    
    def _is_clearly_off_topic(self, ql: str) -> bool:
        journvax_keywords = ['journvax', 'medication', 'side effect', 'dose', 'prescription', 'drug']
        off_topic_keywords = ['bedtime story', 'poem', 'code', 'python', 'javascript', 'weather', 'recipe']
        return any(k in ql for k in off_topic_keywords) and not any(k in ql for k in journvax_keywords)
    
    def _is_potentially_unsafe(self, ql: str) -> bool:
        ambiguous_terms = ['safe', 'okay', 'can i', 'should i', 'what if', 'is it bad']
        return any(t in ql for t in ambiguous_terms)
    
    async def _llm_assess_query(self, query: str) -> Optional[ValidationDecision]:
        try:
            from llm_client import call_huggingface_with_retry
            from config import GUARD_MODEL_PARAMS
            
            prompt = f"""Assess if this pharmaceutical query is safe to answer.

Query: "{query}"

Rules:
- General reminders like "contact your healthcare provider" are ALLOWED and are NOT medical advice.
- Only personalized treatment/dose recommendations count as medical advice.

Respond ONLY with JSON:
{{
  "is_safe": true/false,
  "violation_type": "none|medical_advice|dosing|off_label|misuse",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
            response = await call_huggingface_with_retry(prompt, GUARD_MODEL_PARAMS)
            try:
                m = re.search(r'\{.*\}', response, re.DOTALL)
                if m:
                    res = json.loads(m.group())
                    if not res.get('is_safe', True) and res.get('confidence', 0) > self.llm_safety_threshold:
                        vmap = {
                            'medical_advice': RegulatoryViolation.MEDICAL_ADVICE,
                            'dosing': RegulatoryViolation.UNAPPROVED_DOSING,
                            'off_label': RegulatoryViolation.OFF_LABEL_USE,
                            'misuse': RegulatoryViolation.ADMIN_MISUSE
                        }
                        viol = vmap.get(res.get('violation_type', 'medical_advice'), RegulatoryViolation.MEDICAL_ADVICE)
                        logger.info(f"LLM flagged query: {res.get('reasoning','')}")
                        return self._create_redirect(viol, f"LLM assessment: {res.get('reasoning','Unsafe query')}")
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response as JSON")
        except Exception as e:
            logger.error(f"LLM query assessment failed: {e}")
        return None
    
    # ========================================================================
    # RESPONSE VALIDATION (Embeddings + Patterns + Optional LLM)
    # ========================================================================
    
    async def validate_response(self, response: str, context: str = "", query: str = "", **kwargs) -> ValidationDecision:
        if not self.enabled:
            return ValidationDecision(ValidationResult.APPROVED, response, "Validation disabled", confidence=1.0)
        
        # Skip validation for clear refusals
        if response.lower().startswith(("i cannot", "i don't have", "i'm not able", "i'm sorry")):
            return ValidationDecision(ValidationResult.APPROVED, response, "Refusal response - no validation needed", confidence=0.95)
        
        # 1) Deterministic compliance checks (may auto-correct)
        compliance = self._check_compliance_patterns(response, query)
        if compliance.result != ValidationResult.APPROVED:
            return compliance
        
        # 2) Grounding check
        if context and len(context) > 50 and self.embedding_model is not None:
            gr = self._check_grounding(response, context)
            if gr.result != ValidationResult.APPROVED:
                return gr
            compliance.grounding_score = gr.grounding_score
        
        # 3) LLM safety assessment (final pass; never surface suggestions)
        if self.use_llm_guard:
            llm = await self._llm_assess_response(response, context, query)
            if llm and llm.result != ValidationResult.APPROVED:
                return llm
            if llm:
                compliance.llm_safety_score = llm.confidence
        
        return compliance
    
    def _check_compliance_patterns(self, response: str, query: str) -> ValidationDecision:
        rl = response.lower()
        
        # Allow generic “contact your healthcare provider” statements (not medical advice)
        # Category 1: implied safety from absence
        if re.search(r"(?:doesn't|don't) mention.*(?:so|therefore).*(?:fine|safe|ok)", rl):
            return self._create_rejection(RegulatoryViolation.INACCURATE_CLAIMS, "Implied safety from absence")
        
        # Category 2: risk comm → auto-append disclaimer if listing AEs
        listing_indicators = ['include', 'such as', 'may cause', 'can cause', 'reported', 'common side effects are']
        has_medical_list = any(ind in rl for ind in listing_indicators)
        has_disclaimer = any(p in rl for p in ["not a complete list", "medication guide", "complete safety information", "see the medication guide"])
        if has_medical_list and not has_disclaimer:
            corrected = self._append_disclaimer(response)
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=corrected,
                reasoning="Added missing side-effect scope disclaimer",
                violation=RegulatoryViolation.INADEQUATE_RISK_COMM
            )
        
        # Category 3: off-label use recommendations
        off_label_patterns = [
            r"(?:give|administer|safe to give).*(?:to|for).*(?:child|pediatric|infant|baby)",
            r"(?:child|baby|infant).*(?:can|should|may).*(?:take|use|have)\s+(?:journvax|it|this)",
            r"(?:safe|okay|fine).*(?:during|while|if).*(?:pregnant|pregnancy|nursing|breastfeed)",
            r"use (?:journvax|it|this).*(?:for|to treat).*(?:off-label|unapproved)"
        ]
        for pat in off_label_patterns:
            if re.search(pat, rl):
                return self._create_rejection(RegulatoryViolation.OFF_LABEL_USE, "Off-label use recommendation")
        
        # Category 4: improper promotion (tone)
        problematic_tone = [
            "don't worry", "no need to worry", "should be fine", "nothing to worry about",
            "typically safe", "generally safe", "usually safe", "perfectly safe",
            "err on the side of caution", "better safe than sorry"
        ]
        for phrase in problematic_tone:
            if phrase in rl:
                return self._create_rejection(RegulatoryViolation.IMPROPER_PROMOTION, f"Inappropriate tone: {phrase}")
        
        # Category 5: cross-product references
        if re.search(r"(?:like|similar to|same as|compared to|versus|better than|worse than)\s+(?:other|another|different|competitor)", rl):
            return self._create_rejection(RegulatoryViolation.CROSS_PRODUCT_REF, "Product comparison")
        
        # Category 6: individualized medical advice
        med_advice = [
            r"you (?:should|must|need to) (?:take|stop|start|increase|decrease)",
            r"(?:increase|decrease|double|halve) your (?:dose|dosage)",
            r"your doctor (?:will|should) (?:prescribe|adjust)",
            r"(?:try|consider) (?:taking|stopping|changing) your",
            r"i (?:recommend|suggest|advise) (?:you|that you)"
        ]
        for pat in med_advice:
            if re.search(pat, rl):
                return self._create_rejection(RegulatoryViolation.MEDICAL_ADVICE, "Individual medical guidance")
        
        # Category 7: safety-critical escalation → auto-append emergency sentence, not refuse
        severe_signals = [
            r"trouble breathing",
            r"short(?:ness)? of breath",
            r"swelling (?:of|in) (?:face|lips|tongue|throat)",
            r"faint(?:ing)?|pass\s*out",
            r"chest pain",
            r"anaphylaxis|anaphylactic"
        ]
        has_red_flag = any(re.search(p, rl) for p in severe_signals)
        if has_red_flag and not self._has_emergency_guidance(rl):
            corrected = self._append_emergency_guidance(response)
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=corrected,
                reasoning="Added emergency guidance for red-flag symptoms",
                violation=RegulatoryViolation.SAFETY_CRITICAL_MISS
            )
        
        # Category 8: administration misuse
        admin_misuse = [
            r"(?:share|split|crush|chew|inject|snort)\s+(?:the|your|journvax)",
            r"give (?:it|journvax) to (?:someone|friend|family)"
        ]
        for pat in admin_misuse:
            if re.search(pat, rl):
                return self._create_rejection(RegulatoryViolation.ADMIN_MISUSE, "Unsafe administration")
        
        # Category 9: dosing guidance not in label
        dosing_pats = [
            r"take \d+\s*(?:tablet|pill|mg|ml|capsule)",
            r"take (?:it|journvax) (?:in the |at )?(?:morning|evening|night|bedtime)",
            r"take (?:with|without|before|after) (?:food|meal|eating)",
            r"(?:stick to|maintain|follow) (?:a |your |the )?(?:regular|usual) schedule"
        ]
        for pat in dosing_pats:
            if re.search(pat, rl) and not re.search(r'(according to|per|as stated in|medication guide says)', rl):
                return self._create_rejection(RegulatoryViolation.UNAPPROVED_DOSING, "Dosing guidance")
        
        return ValidationDecision(ValidationResult.APPROVED, response, "Pattern compliance check passed", confidence=0.95)
    
    def _check_grounding(self, response: str, context: str) -> ValidationDecision:
        try:
            resp = self.embedding_model.encode(response, show_progress_bar=False)
            ctx = self.embedding_model.encode(context, show_progress_bar=False)
            sim = float(np.dot(resp, ctx) / (np.linalg.norm(resp) * np.linalg.norm(ctx)))
            unsupported = self._find_unsupported_claims(response, context)
            if sim < self.similarity_threshold and len(unsupported) > 2:
                logger.warning(f"Poor grounding: {sim:.2f}, unsupported: {len(unsupported)}")
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=REFUSAL_TEMPLATES["default"],
                    reasoning=f"Poor grounding (score: {sim:.2f})",
                    grounding_score=sim,
                    unsupported_claims=unsupported,
                    violation=RegulatoryViolation.INACCURATE_CLAIMS
                )
            return ValidationDecision(ValidationResult.APPROVED, response, "Grounding check passed", grounding_score=sim)
        except Exception as e:
            logger.error(f"Grounding check failed: {e}")
            return ValidationDecision(ValidationResult.APPROVED, response, "Grounding check error", grounding_score=0.5)
    
    async def _llm_assess_response(self, response: str, context: str, query: str) -> Optional[ValidationDecision]:
        """Use LLM for nuanced safety assessment. Never surface LLM 'suggestions' to the user."""
        try:
            from llm_client import call_huggingface_with_retry
            from config import GUARD_MODEL_PARAMS
            
            prompt = f"""Assess this pharmaceutical response for safety violations.

User Query: "{query}"
Response: "{response[:500]}"
Context Available: {"Yes" if context else "No"}

Important rules:
- General statements like "contact your healthcare provider/pharmacist" are ALLOWED and are NOT medical advice.
- Flag only personalized treatment decisions or dosing recommendations as medical advice.
- Return ONLY JSON; do NOT include editorial text meant for end users.

JSON schema:
{{
  "is_safe": true/false,
  "violation": "none|medical_advice|dosing|improper_tone|ungrounded|off_label",
  "confidence": 0.0-1.0,
  "specific_issue": "what exactly is problematic"
}}"""
            llm = await call_huggingface_with_retry(prompt, GUARD_MODEL_PARAMS)
            try:
                m = re.search(r'\{.*\}', llm, re.DOTALL)
                if not m:
                    return None
                res = json.loads(m.group())
                if res.get("is_safe", True) or res.get("confidence", 0) <= self.llm_safety_threshold:
                    return None
                
                v = res.get("violation", "medical_advice")
                if v in ("medical_advice", "dosing", "off_label"):
                    return self._create_rejection(
                        {
                            "medical_advice": RegulatoryViolation.MEDICAL_ADVICE,
                            "dosing": RegulatoryViolation.UNAPPROVED_DOSING,
                            "off_label": RegulatoryViolation.OFF_LABEL_USE
                        }[v],
                        f"LLM: {res.get('specific_issue','Safety violation')}"
                    )
                elif v == "ungrounded":
                    return self._create_rejection(RegulatoryViolation.INACCURATE_CLAIMS, f"LLM: {res.get('specific_issue','Ungrounded claim')}")
                elif v == "improper_tone":
                    # Replace with neutral refusal rather than surfacing suggestions
                    return self._create_rejection(RegulatoryViolation.IMPROPER_PROMOTION, f"LLM: {res.get('specific_issue','Improper tone')}")
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM safety assessment")
        except Exception as e:
            logger.error(f"LLM response assessment failed: {e}")
        return None
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _append_disclaimer(self, text: str) -> str:
        text = text.rstrip().rstrip('.')
        return f"{text}. This is not a complete list. See the Medication Guide for full information."
    
    def _has_emergency_guidance(self, rl: str) -> bool:
        return bool(re.search(r"(seek|get)\s+(immediate|emergency)\s+(medical\s+)?(help|care)|call\s+911", rl))
    
    def _append_emergency_guidance(self, text: str) -> str:
        text = text.rstrip()
        if not text.endswith('.'):
            text += '.'
        emergency = " If you have trouble breathing, swelling of the face, lips, tongue, or throat, fainting, chest pain, or signs of a severe allergic reaction, seek emergency medical care immediately."
        return text + emergency
    
    def _create_redirect(self, violation: RegulatoryViolation, reason: str) -> ValidationDecision:
        return ValidationDecision(
            result=ValidationResult.REDIRECT,
            final_response=REFUSAL_TEMPLATES.get(violation, REFUSAL_TEMPLATES["default"]),
            reasoning=reason,
            confidence=0.95,
            violation=violation,
            should_log=True
        )
    
    def _create_rejection(self, violation: RegulatoryViolation, reason: str) -> ValidationDecision:
        return ValidationDecision(
            result=ValidationResult.REJECTED,
            final_response=REFUSAL_TEMPLATES.get(violation, REFUSAL_TEMPLATES["default"]),
            reasoning=reason,
            confidence=0.90,
            violation=violation,
            should_log=True
        )
    
    def _find_unsupported_claims(self, response: str, context: str) -> List[str]:
        unsupported: List[str] = []
        sentences = re.split(r'[.!?]+', response)
        ctx = context.lower()
        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue
            skip = [
                "i don't", "i cannot", "please consult", "according to",
                "medication guide", "healthcare provider", "this is not",
                "see the", "for full information", "journvax"
            ]
            if any(p in s.lower() for p in skip):
                continue
            specific = re.findall(r'\b(?:avoid|take with|do not take with|limit)\s+(\w+)', s.lower())
            for item in specific:
                if item not in ctx and item not in ['water', 'food', 'it', 'medication', 'this', 'that']:
                    unsupported.append(s); break
            numbers = re.findall(r'\b\d+\s*(?:mg|ml|percent|%)\b', s.lower())
            for num in numbers:
                if num not in ctx:
                    unsupported.append(s); break
        return unsupported
    
    def get_validation_summary(self, decision: ValidationDecision) -> Dict:
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

hybrid_guard = HybridSafetyGuard()
unified_guard = hybrid_guard
enhanced_guard = hybrid_guard
persona_validator = hybrid_guard
simple_guard = hybrid_guard

def evaluate_response(context: str, user_question: str, assistant_response: str, **kwargs) -> Tuple[bool, str, str]:
    """Legacy compatibility function"""
    import asyncio
    async def validate():
        return await hybrid_guard.validate_response(response=assistant_response, context=context, query=user_question)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(asyncio.run, validate())
                result = fut.result()
        else:
            result = loop.run_until_complete(validate())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(validate())
    is_safe = (result.result == ValidationResult.APPROVED)
    return is_safe, result.final_response, result.reasoning
