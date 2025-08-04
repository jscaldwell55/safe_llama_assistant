import logging
from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DEFAULT_FALLBACK_MESSAGE, ENABLE_GUARD, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedGuardAgent:
    """
    A simplified guard agent that focuses on factual grounding and safety.
    """
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_messages = {
            "default": "I'm sorry, I can't provide that information. Please try asking in a different way.",
            "ungrounded": "I can't answer that because the information isn't in my knowledge base. Is there another topic I can help with?",
            "unsafe_medical": "For your safety, I cannot provide medical advice or recommendations. Please consult with a qualified healthcare professional.",
        }
        self.embedding_model = None
        if self.enabled:
            self._init_embedding_model()

    def _init_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Guard's embedding model initialized: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize guard's embedding model: {e}", exc_info=True)

    def evaluate(self, context: str, assistant_response: str) -> Tuple[bool, str, str]:
        """Evaluates a response for grounding and safety violations."""
        if not self.enabled:
            return True, assistant_response, "Guard disabled"

        try:
            violations = []
            claims = self._extract_factual_claims(assistant_response)
            
            violations.extend(self._check_grounding(claims, context))
            violations.extend(self._check_safety(assistant_response))

            if not violations:
                logger.info("Guard approved response.")
                return True, assistant_response, "Approved: Response is grounded and safe."
            else:
                reasoning = "; ".join([v['description'] for v in violations])
                logger.warning(f"Guard rejected response: {reasoning}")
                fallback = self._get_appropriate_fallback(violations)
                return False, fallback, reasoning
        except Exception as e:
            logger.error(f"Guard evaluation failed unexpectedly: {e}", exc_info=True)
            return False, self.fallback_messages["default"], str(e)

    def _check_grounding(self, claims: List[str], context: str) -> List[Dict[str, Any]]:
        violations = []
        if not context and claims:
            return [{"type": "ungrounded_claim", "description": "Response has claims but no context was provided."}]
        for claim in claims:
            score = self._calculate_grounding_score(claim, context)
            if score < 0.7:
                violations.append({"type": "ungrounded_claim", "description": f"Ungrounded claim (score: {score:.2f}): '{claim[:50]}...'"})
        return violations

    def _check_safety(self, response: str) -> List[Dict[str, Any]]:
        violations = []
        directive_patterns = [r'\byou (should|must|need to)\b', r'\b(start|stop|increase|decrease) (taking|your dose)\b']
        for pattern in directive_patterns:
            if re.search(pattern, response, re.IGNORECASE) and "according to the documentation" not in response.lower():
                violations.append({"type": "medical_advice", "description": "Contains a medical directive."})
                break
        return violations

    def _extract_factual_claims(self, response: str) -> List[str]:
        sentences = re.split(r'[.!?]+', response)
        claims = []
        non_claim_patterns = [r'^(hello|hi|hey|thanks|thank you)', r'^\s*(i\'m sorry|i don\'t|i can\'t|i couldn\'t find)', r'^\s*(would you like|do you have|can i help)']
        for s in sentences:
            s = s.strip()
            if not s: continue
            is_filler = any(re.search(p, s, re.IGNORECASE) for p in non_claim_patterns)
            if not is_filler and len(s.split()) > 3:
                claims.append(s)
        return claims

    def _calculate_grounding_score(self, statement: str, context: str) -> float:
        if not self.embedding_model or not context: return 0.0
        try:
            statement_embedding = self.embedding_model.encode(statement)
            context_sentences = re.split(r'[.!?]+', context)
            context_embeddings = self.embedding_model.encode([s for s in context_sentences if s.strip()])
            if len(context_embeddings) == 0: return 0.0
            sims = np.dot(context_embeddings, statement_embedding) / (np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(statement_embedding))
            return float(np.max(sims))
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}", exc_info=True)
            return 0.0
            
    def _get_appropriate_fallback(self, violations: List[Dict[str, Any]]) -> str:
        if any(v['type'] == 'medical_advice' for v in violations):
            return self.fallback_messages["unsafe_medical"]
        return self.fallback_messages["ungrounded"]

# --- LAZY-LOADING FUNCTION and PUBLIC API ---
_guard_agent_instance = None
def get_guard_agent():
    """Lazy-loads and returns a single instance of the SimplifiedGuardAgent."""
    global _guard_agent_instance
    if _guard_agent_instance is None:
        _guard_agent_instance = SimplifiedGuardAgent()
    return _guard_agent_instance

def evaluate_response(context: str, assistant_response: str) -> Tuple[bool, str, str]:
    """Public API function for app.py to call."""
    agent = get_guard_agent()
    return agent.evaluate(context, assistant_response)
