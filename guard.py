import logging
from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np
from config import (
    DEFAULT_FALLBACK_MESSAGE,
    ENABLE_GUARD,
    EMBEDDING_MODEL_NAME,
    SEMANTIC_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# --- Embedding model (shared if available) ---
def _load_embedding_model():
    try:
        from embeddings import get_embedding_model  # shared singleton
        return get_embedding_model()
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Guard could not init embedding model: {e}")
            return None

# --- Patterns for critical checks ---
MISUSE_PATTERNS = [
    r"\bcrush(?:ing)?\b", r"\bsnort(?:ing|ed)?\b", r"\binject(?:ing|ed|ion)?\b",
    r"\bsmok(?:e|ing|ed)\b", r"\bplug(?:ging)?\b", r"\bboof(?:ing)?\b", r"\bparachut(?:e|ing)\b",
    r"\biv\b", r"\bdissolv(?:e|ing)\b.*\b(in|into)\b.*\b(injection|needle|vein)\b",
    r"\bchew(?:ing)?\b.*\btablet\b", r"\bmix(?:ing)?\b.*\b(alcohol|vodka|beer|wine)\b",
]
OFFLABEL_CUES = [
    r"\boff[- ]label\b", r"\bnot approved\b", r"\bunapproved\b", r"\bnot indicated\b",
    r"\bfor (?:adhd|weight|sleep|pain|anxiety in children|performance)\b",
]
PROMO_PATTERNS = [
    r"\bbest\b.*\b(drug|medication|treatment)\b", r"\bmost effective\b", r"\bguaranteed\b",
    r"\bamazing results\b", r"\bbreakthrough\b", r"\bmiraculous\b", r"\brevolutionary\b",
]
DOSING_DIRECTIVES = [
    r'\byou (should|must|need to) take\b', r'\bstart (taking|with)\b.*\bmg\b',
    r'\bstop taking\b', r'\bincrease your dose\b', r'\bdecrease your dose\b',
    r'\btake \d+\s*mg\b', r'\bdo not take\b.*\bif you\b'
]
COMPETITOR_CUES = [
    r"\bversus\b", r"\bvs\.\b", r"\bcompared to\b", r"\bunlike\b"
]

class EnhancedGuardAgent:
    """
    Binary guard:
      1) Grounding to RAG context (semantic + lexical).
      2) Enforce 5 Critical Safety Requirements with immediate reject on violation.
    Heuristic-first. No LLM verdicts (prevents hallucinated rejections).
    """

    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.embedding_model = _load_embedding_model()
        try:
            self.sim_threshold = float(SEMANTIC_SIMILARITY_THRESHOLD)
        except Exception:
            self.sim_threshold = 0.65  # sensible default

        # User-facing fallbacks
        self.fallback_messages = {
            "default": "I can't help with that. Can we talk about something else?",
            "no_context": "I don't have sufficient information in our knowledge base to answer that. Could you rephrase your question or ask about something else?",
            "unsafe_medical": "I cannot provide medical advice. Please consult with a healthcare professional for medical guidance.",
            "off_label_refusal": "I can’t provide information about unapproved or off-label uses. I can share what’s documented in the approved labeling.",
            "misuse_refusal": (
                "I can’t help with questions about misusing or altering medications. "
                "If you’re in immediate danger, call local emergency services. "
                "For confidential support with substance use, in the U.S. you can call SAMHSA at 1-800-662-HELP (4357)."
            ),
        }

    # ---------- Public API ----------
    def evaluate_response(
        self,
        context: str,
        user_question: str,
        assistant_response: str,
        conversation_history: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        if not self.enabled:
            return True, assistant_response, "Guard disabled"
        try:
            # 0) Critical pre-checks driven by the USER QUESTION
            if self._is_misuse_intent(user_question):
                return False, self.fallback_messages["misuse_refusal"], "Reject: misuse/abusive route intent in user question."

            if self._is_offlabel_probe(user_question) and not self._has_safe_offlabel_context(context):
                return False, self.fallback_messages["off_label_refusal"], "Reject: off-label request without explicit 'not indicated' context."

            # 1) Intent (for logging only)
            intent = self._analyze_response_intent(assistant_response)
            logger.info(f"Identified response intent: {intent}")

            # 2) Extract factual claims
            claims = self._extract_factual_claims(assistant_response)

            # 3) Grounding gate: factual content but no context → reject
            if claims and not (context and context.strip()):
                return False, self.fallback_messages["no_context"], "Reject: factual content with no retrieved context."

            # 4) Safety checks (5 Critical Requirements)
            safety_violations = self._safety_violations(user_question, assistant_response, context)
            if safety_violations:
                fallback = self._fallback_for_violations(safety_violations)
                return False, fallback, f"Reject: {safety_violations[0]['description']}"

            # 5) Per-claim grounding (semantic + lexical)
            grounding_violations = []
            for claim in claims:
                score = self._semantic_similarity(claim, context)
                if score < self.sim_threshold and not self._lexically_grounded(claim, context):
                    grounding_violations.append({
                        "type": "ungrounded_claim",
                        "severity": "high",
                        "claim": claim,
                        "score": score,
                        "description": f"Claim not supported by context (score={score:.2f})"
                    })
            if grounding_violations:
                return False, self.fallback_messages["no_context"], "Reject: ungrounded factual claim(s)."

            # APPROVE
            return True, assistant_response, f"Approve: grounded and safety-compliant (intent={intent})."

        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}", exc_info=True)
            return False, self.fallback_messages["default"], f"Guard evaluation error: {str(e)}"

    # ---------- Critical requirement checks ----------
    def _safety_violations(self, question: str, response: str, context: str) -> List[Dict[str, Any]]:
        v: List[Dict[str, Any]] = []
        rl = response.lower()

        # 1) Medical advice / treatment recommendations (beyond documented)
        if self._contains_pattern(rl, DOSING_DIRECTIVES):
            v.append({"type": "medical_advice", "severity": "critical", "description": "Directive medical language."})

        # 2) Dosage instructions not explicitly in context
        if re.search(r'\b\d+\s*mg\b', rl) and not self._lexically_grounded(response, context):
            v.append({"type": "dosage_not_in_context", "severity": "critical", "description": "Dosage mentioned without explicit support in context."})

        # 3) Misuse / unapproved route
        if self._is_misuse_intent(question) or self._contains_pattern(rl, MISUSE_PATTERNS):
            v.append({"type": "misuse", "severity": "critical", "description": "Misuse or unapproved route of administration."})
        if self._is_offlabel_probe(question) and not self._has_safe_offlabel_context(context):
            v.append({"type": "off_label", "severity": "high", "description": "Off-label topic without explicit 'not indicated' support."})

        # 4) Competitor mentions not in context (rough heuristic)
        if self._contains_pattern(rl, COMPETITOR_CUES):
            brandish = set(re.findall(r'\b[A-Z][a-zA-Z0-9]{2,}\b', response))
            brandish = {b for b in brandish if b.lower() not in {"i", "we", "assistant", "user"}}
            for b in brandish:
                if b.lower() not in context.lower():
                    v.append({"type": "competitor_mention", "severity": "medium", "description": f"Competitor/product mention not present in context: {b}"})
                    break

        # 5) Promotional language
        if self._contains_pattern(rl, PROMO_PATTERNS):
            v.append({"type": "promotional_language", "severity": "medium", "description": "Promotional or exaggerated wording."})

        return v

    # ---------- Intent detection (logging only) ----------
    def _analyze_response_intent(self, response: str) -> str:
        if self._has_factual_content(response):
            return "ANSWERING"
        rl = response.lower()
        if any(p in rl for p in ["could you clarify", "could you specify", "which aspect", "do you mean", "are you asking about"]):
            return "CLARIFYING"
        if any(p in rl for p in ["i don't have information", "not in the documentation", "unable to find"]):
            return "ACKNOWLEDGING_GAP"
        if any(p in rl for p in ["i can help with", "would you like to know about", "alternatively"]):
            return "OFFERING_ALTERNATIVES"
        return "CONVERSATIONAL_BRIDGE"

    # ---------- Factual claim extraction ----------
    def _extract_factual_claims(self, response: str) -> List[str]:
        if not response:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s.strip() for s in sentences if s and len(s.strip()) > 3]

        claims: List[str] = []
        conversational_exclusions = [
            r'^(hello|hi|hey)\b',
            r"^(thanks|thank you)\b",
            r"^(i'?m sorry|i do(?: not|n't) have|i can(?:not|\'t))\b",
            r"(?:not in (?:the )?documentation|don'?t have information|unable to find|no information)",
        ]
        factual_indicators = [
            r'\b(?:is|are|was|were|be)\s+\w+',
            r'\b(?:contains?|includes?|has|have)\s+\w+',
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+',
            r'\b(?:works?\s+by|functions?\s+through)\s+\w+',
            r'\b\d+\s*(?:percent|%|mg|ml|mcg|iu|units?)\b',
            r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',
        ]
        for s in sentences:
            sl = s.lower()
            if any(re.search(p, sl) for p in conversational_exclusions):
                continue
            if any(re.search(p, sl) for p in factual_indicators):
                claims.append(s)
        return claims

    # ---------- Helpers: misuse/off-label ----------
    def _is_misuse_intent(self, text: str) -> bool:
        return self._contains_pattern(text.lower(), MISUSE_PATTERNS)

    def _is_offlabel_probe(self, text: str) -> bool:
        tl = text.lower()
        return any(re.search(p, tl) for p in OFFLABEL_CUES)

    def _has_safe_offlabel_context(self, context: str) -> bool:
        if not context:
            return False
        return bool(re.search(r"\b(not indicated|not approved)\b", context.lower()))

    # ---------- Helpers: grounding ----------
    def _has_factual_content(self, text: str) -> bool:
        indicators = [
            r'\b(?:is|are|was|were)\s+\w+', r'\b(?:contains?|includes?|has|have)\s+\w+',
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+', r'\b(?:works?\s+by|functions?\s+through)\s+\w+',
            r'\b\d+\s*(?:percent|%|mg|ml|mcg|iu|units?)\b', r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',
        ]
        return any(re.search(p, text.lower()) for p in indicators)

    def _semantic_similarity(self, statement: str, context: str) -> float:
        if not context:
            return 0.0
        if not self.embedding_model:
            return 1.0 if self._lexically_grounded(statement, context) else 0.0
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
            return 1.0 if self._lexically_grounded(statement, context) else 0.0

    def _lexically_grounded(self, statement: str, context: str) -> bool:
        if not context:
            return False
        claim_words = set(re.findall(r'\b\w+\b', statement.lower()))
        ctx_words = set(re.findall(r'\b\w+\b', context.lower()))
        stop = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','that','this','it','as'}
        claim_words -= stop
        ctx_words -= stop
        if not claim_words:
            return True
        overlap = len(claim_words & ctx_words) / len(claim_words)
        return overlap > 0.50

    def _contains_pattern(self, text_lower: str, patterns: List[str]) -> bool:
        return any(re.search(p, text_lower) for p in patterns)

    def _fallback_for_violations(self, violations: List[Dict[str, Any]]) -> str:
        types = [v.get("type") for v in violations]
        if "misuse" in types:
            return self.fallback_messages["misuse_refusal"]
        if "off_label" in types:
            return self.fallback_messages["off_label_refusal"]
        if "medical_advice" in types or "dosage_not_in_context" in types:
            return self.fallback_messages["unsafe_medical"]
        if "ungrounded_claim" in types:
            return self.fallback_messages["no_context"]
        return self.fallback_messages["default"]

# Global instance + wrapper
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
