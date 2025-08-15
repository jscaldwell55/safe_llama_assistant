# guard.py
import logging
from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np

# We don't actually need the formatted guard prompt text at runtime here,
# so we avoid importing prompts to reduce hot-reload churn.
from config import (
    DEFAULT_FALLBACK_MESSAGE,
    ENABLE_GUARD,
    EMBEDDING_MODEL_NAME,
    SEMANTIC_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# --- Embedding model access ---
def _load_embedding_model():
    """
    Try to use the shared embeddings loader if present; otherwise load locally.
    Returns None on failure (guard will degrade gracefully).
    """
    try:
        # Preferred: shared singleton loader (emits logs like "embeddings - INFO - Loaded ...")
        from embeddings import get_embedding_model  # type: ignore
        return get_embedding_model()
    except Exception:
        # Fallback: direct load (avoid noisy logs on Streamlit hot-reload)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            return SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Guard could not init embedding model: {e}")
            return None


class EnhancedGuardAgent:
    """
    Intent-aware guard agent that evaluates responses based on conversational context and safety requirements.
    Binary approve/reject with context grounding + medical safety checks.
    """

    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_messages = {
            "default": "I can't help with that. Can we talk about something else?",
            "no_context": "I don't have sufficient information in our knowledge base to answer that. Could you rephrase your question or ask about something else?",
            "unsafe_medical": "I cannot provide medical advice. Please consult with a healthcare professional for medical guidance.",
            "no_info": "I'm sorry, I don't seem to have any information on that topic in our documentation. Can I help you with something else?",
        }
        self.embedding_model = _load_embedding_model()
        # Use configured threshold; if unset, default to a slightly more permissive 0.60 to reduce false rejects.
        try:
            self.sim_threshold = float(SEMANTIC_SIMILARITY_THRESHOLD)
        except Exception:
            self.sim_threshold = 0.60

    # ---------- Public API ----------
    def evaluate_response(
        self,
        context: str,
        user_question: str,
        assistant_response: str,
        conversation_history: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        Returns: (is_approved, final_response, guard_reasoning)
        """
        if not self.enabled:
            return True, assistant_response, "Guard disabled"

        try:
            intent = self._analyze_response_intent(assistant_response)
            logger.info(f"Identified response intent: {intent}")

            claims = self._extract_claims_by_intent(assistant_response, intent)
            violations = self._check_violations_by_intent(
                assistant_response, context, claims, intent
            )
            verdict = self._make_verdict(violations, intent)

            if verdict["verdict"] == "APPROVE":
                return True, assistant_response, verdict["reasoning"]

            # REJECT path → choose appropriate fallback
            fallback = self._get_appropriate_fallback(verdict.get("violations", []))
            return False, fallback, verdict["reasoning"]

        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}", exc_info=True)
            return False, self.fallback_messages["default"], f"Guard evaluation error: {str(e)}"

    # ---------- Intent detection ----------
    def _has_factual_content(self, text: str) -> bool:
        indicators = [
            r'\b(?:is|are|was|were)\s+\w+',
            r'\b(?:contains?|includes?|has|have)\s+\w+',
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+',
            r'\b(?:works?\s+by|functions?\s+through)\s+\w+',
            r'\b\d+\s*(?:percent|%|mg|ml|mcg|iu|units?)\b',
            r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',
        ]
        tl = text.lower()
        return any(re.search(p, tl) for p in indicators)

    def _analyze_response_intent(self, response: str) -> str:
        rl = response.lower()

        # If there is any factual content, treat as ANSWERING (even if it starts with a greeting)
        if self._has_factual_content(response):
            return "ANSWERING"

        # Clarifying (no facts)
        if any(
            p in rl
            for p in ["could you clarify", "could you specify", "which aspect", "do you mean", "are you asking about"]
        ):
            return "CLARIFYING"

        # Acknowledging gap (no facts)
        if any(
            p in rl
            for p in [
                "don't have information",
                "don't have specific",
                "no information available",
                "not in our documentation",
                "not in the provided",
                "unable to find",
                "don't see any mention",
                "doesn't appear to be documented",
            ]
        ):
            return "ACKNOWLEDGING_GAP"

        # Alternatives (no facts)
        if any(
            p in rl
            for p in [
                "i can help with",
                "i can provide information about",
                "would you like to know about",
                "i can share information on",
                "related topics include",
                "alternatively",
            ]
        ):
            return "OFFERING_ALTERNATIVES"

        # Otherwise, conversational bridge
        return "CONVERSATIONAL_BRIDGE"

    # ---------- Claim extraction ----------
    def _extract_claims_by_intent(self, response: str, intent: str) -> List[Dict[str, Any]]:
        if intent in ["CONVERSATIONAL_BRIDGE", "CLARIFYING"]:
            return []
        if intent == "ACKNOWLEDGING_GAP":
            return self._extract_alternative_claims(response)
        return self._extract_factual_claims(response)

    def _extract_alternative_claims(self, response: str) -> List[Dict[str, Any]]:
        claims = []
        rl = response.lower()
        patterns = [
            r"i can (provide|share|help with) information about ([^.]+)",
            r"our documentation (includes|covers|contains) ([^.]+)",
            r"available information includes ([^.]+)",
            r"we have (information|documentation) on ([^.]+)",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, rl):
                claims.append({"text": m.group(0), "type": "available_info", "requires_grounding": True})
        return claims

    def _extract_factual_claims(self, response: str) -> List[Dict[str, Any]]:
        sentences = re.split(r'[.!?]+', response)
        claims = []
        indicators = [
            r'\b(?:is|are|was|were)\s+\w+',
            r'\b(?:contains?|includes?|has|have)\s+\w+',
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+',
            r'\b(?:works?\s+by|functions?\s+through)\s+\w+',
            r'\b\d+\s*(?:percent|%|mg|ml|mcg|iu|units?)\b',
            r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',
        ]
        conversational = [
            r'^(?:hello|hi|hey|thanks|thank you)',
            r'^\s*(?:i\'m sorry|i don\'t|i can\'t)',
            r'^\s*(?:would you like|do you have)',
            r'^\s*(?:based on|according to)',
            r'don\'t have (?:specific |any )?information',
            r'(?:not |don\'t see any )(?:mention|information|details)',
            r'unable to (?:find|provide)',
            r'(?:not |doesn\'t appear to be )(?:in|documented)',
            r'no (?:specific |documented )?(?:information|details|data)',
            r'knowledge base (?:doesn\'t|does not) (?:contain|include|have)',
        ]
        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            if any(re.search(p, s.lower()) for p in conversational):
                continue
            if any(re.search(p, s.lower()) for p in indicators):
                claims.append({"text": s, "type": "factual", "requires_grounding": True})
        return claims

    # ---------- Violation checks ----------
    def _check_violations_by_intent(
        self, response: str, context: str, claims: List[Dict[str, Any]], intent: str
    ) -> List[Dict[str, Any]]:
        violations = []

        # ANSWERING or OFFERING_ALTERNATIVES without context is not allowed
        if intent in ["ANSWERING", "OFFERING_ALTERNATIVES"] and not (context and context.strip()):
            return [
                {
                    "type": "ungrounded_claim",
                    "severity": "high",
                    "description": "Factual response attempted without retrieved context",
                }
            ]

        if intent == "CONVERSATIONAL_BRIDGE":
            if self._contains_medical_directive(response):
                violations.append(
                    {"type": "medical_advice", "severity": "critical", "description": "Medical directive in conversational response"}
                )
            return violations

        if intent == "ACKNOWLEDGING_GAP":
            for claim in claims:
                if claim.get("requires_grounding") and not self._is_claim_grounded_lexical(claim["text"], context):
                    violations.append(
                        {"type": "ungrounded_claim", "severity": "high", "description": f"Claims availability not in context: {claim['text']}"}
                    )
            return violations

        if intent in ["ANSWERING", "OFFERING_ALTERNATIVES"]:
            violations.extend(self._validate_factual_response(response, context, claims))
            return violations

        if intent == "CLARIFYING":
            if self._contains_medical_directive(response):
                violations.append(
                    {"type": "medical_advice", "severity": "critical", "description": "Medical directive in clarifying response"}
                )
            return violations

        return violations

    def _validate_factual_response(self, response: str, context: str, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        violations = []
        for claim in claims:
            score = self._calculate_grounding_score(claim["text"], context)

            # Require BOTH low embedding similarity AND poor lexical overlap before flagging ungrounded.
            if score < self.sim_threshold and not self._is_claim_grounded_lexical(claim["text"], context):
                violations.append(
                    {
                        "type": "ungrounded_claim",
                        "severity": "high",
                        "claim": claim["text"],
                        "score": score,
                        "description": f"Insufficient grounding (score: {score:.2f})",
                    }
                )

        violations.extend(self._check_medical_safety(response, context))
        if self._contains_promotional_language(response):
            violations.append(
                {"type": "promotional_language", "severity": "medium", "description": "Contains promotional or exaggerated claims"}
            )
        return violations

    # ---------- Verdict & fallbacks ----------
    def _make_verdict(self, violations: List[Dict[str, Any]], intent: str) -> Dict[str, Any]:
        if violations:
            order = ["critical", "high", "medium", "low"]
            by = {}
            for v in violations:
                by.setdefault(v.get("severity", "low"), []).append(v)
            most = next((by[s][0] for s in order if s in by and by[s]), violations[0])
            return {
                "verdict": "REJECT",
                "reasoning": f"Safety violation detected: {most['description']}",
                "violations": violations,
                "intent": intent,
            }
        return {"verdict": "APPROVE", "reasoning": f"Response approved - Intent: {intent}", "intent": intent}

    def _get_appropriate_fallback(self, violations: List[Dict[str, Any]]) -> str:
        """Pick a user-facing fallback based on the most severe violation types."""
        types = {v.get("type") for v in violations}
        if {"medical_advice", "off_label"} & types:
            return self.fallback_messages["unsafe_medical"]
        if {"ungrounded_claim", "knowledge_leakage"} & types:
            return self.fallback_messages["no_context"]
        if {"promotional_language"} & types:
            return self.fallback_messages["default"]
        # Default safety net
        return self.fallback_messages["default"]

    # ---------- Helpers ----------
    def _contains_medical_directive(self, response: str) -> bool:
        patterns = [
            r'\byou (should|must|need to) take\b',
            r'\bstart (taking|with)\b.*\bmg\b',
            r'\bstop taking\b',
            r'\bincrease your dose\b',
            r'\bdecrease your dose\b',
            r'\btake \d+\s*mg\b',
            r'\bdo not take\b.*\bif you\b',
        ]
        rl = response.lower()
        for p in patterns:
            if re.search(p, rl) and not any(ind in rl for ind in ["according to", "documentation states", "guidelines say"]):
                return True
        return False

    def _check_medical_safety(self, response: str, context: str) -> List[Dict[str, Any]]:
        violations = []
        rl = response.lower()
        checks = [
            (r'effective for treating', 'efficacy_claim'),
            (r'will help with', 'efficacy_claim'),
            (r'cures?', 'efficacy_claim'),
            (r'safe for everyone', 'safety_overstatement'),
            (r'no side effects', 'safety_overstatement'),
            (r'completely safe', 'safety_overstatement'),
        ]
        for pattern, vtype in checks:
            if re.search(pattern, rl) and not self._is_claim_grounded_lexical(pattern, context):
                violations.append({"type": "medical_advice", "severity": "high", "description": f"Ungrounded {vtype.replace('_', ' ')}"})
        return violations

    def _contains_promotional_language(self, response: str) -> bool:
        patterns = [
            r'\bbest\b.*\b(drug|medication|treatment)\b',
            r'\bmost effective\b',
            r'\bguaranteed\b',
            r'\bamazing results\b',
            r'\bbreakthrough\b',
            r'\bmiraculous\b',
            r'\brevolutionary\b',
        ]
        return any(re.search(p, response.lower()) for p in patterns)

    def _is_claim_grounded_lexical(self, claim: str, context: str) -> bool:
        if not context:
            return False
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        stop = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
        claim_words -= stop
        context_words -= stop
        if not claim_words:
            return True
        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap > 0.50  # slightly permissive to reduce false negatives

    def _calculate_grounding_score(self, statement: str, context: str) -> float:
        """
        Max cosine similarity against context sentences.
        If no embedding model, fall back to lexical grounding as a proxy.
        """
        if not context:
            return 0.0
        if not self.embedding_model:
            return 1.0 if self._is_claim_grounded_lexical(statement, context) else 0.0

        try:
            stmt = self.embedding_model.encode(statement, convert_to_tensor=False, show_progress_bar=False)
            ctx_sentences = [s.strip() for s in re.split(r'[.!?]+', context) if len(s.strip()) > 20]
            if not ctx_sentences:
                ctx = self.embedding_model.encode(context, convert_to_tensor=False, show_progress_bar=False)
                return float(np.dot(stmt, ctx) / (np.linalg.norm(stmt) * np.linalg.norm(ctx)))
            max_sim = 0.0
            for c in ctx_sentences:
                ce = self.embedding_model.encode(c, convert_to_tensor=False, show_progress_bar=False)
                sim = float(np.dot(stmt, ce) / (np.linalg.norm(stmt) * np.linalg.norm(ce)))
                max_sim = max(max_sim, sim)
            return max_sim
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            # degrade gracefully
            return 1.0 if self._is_claim_grounded_lexical(statement, context) else 0.0


# Global guard agent instance & public wrapper
guard_agent = EnhancedGuardAgent()

def evaluate_response(
    context: str,
    user_question: str,
    assistant_response: str,
    conversation_history: Optional[str] = None,
) -> Tuple[bool, str, str]:
    return guard_agent.evaluate_response(context, user_question, assistant_response, conversation_history)

# Legacy helpers
def is_safe_response(text: str) -> bool:
    return len(text.strip()) >= 10

def redirect_response() -> str:
    return DEFAULT_FALLBACK_MESSAGE
