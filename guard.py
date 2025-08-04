# guard.py

import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import re
from config import DEFAULT_FALLBACK_MESSAGE, ENABLE_GUARD, EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedGuardAgent:
    """
    A simplified guard agent that focuses on factual grounding and safety,
    assuming intent has been handled pre-generation.
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_messages = {
            "default": "I'm sorry, I can't provide that information. Please try asking in a different way.",
            "ungrounded": "I can't answer that because the information isn't in my knowledge base. Is there another topic I can help with?",
            "unsafe_medical": "For your safety, I cannot provide medical advice or recommendations. Please consult with a qualified healthcare professional.",
        }
        self.embedding_model = None
        if ENABLE_GUARD:
            self._init_embedding_model()

    def _init_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Guard's embedding model initialized: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize guard's embedding model: {e}")

    def evaluate_response(
        self, 
        context: str, 
        assistant_response: str,
    ) -> Tuple[bool, str, str]:
        """
        Evaluates a response for grounding and safety violations.
        
        Args:
            context (str): The RAG context that was used for generation.
            assistant_response (str): The final generated response.
            
        Returns:
            Tuple[bool, str, str]: (is_approved, final_response, guard_reasoning)
        """
        if not self.enabled:
            return True, assistant_response, "Guard disabled"

        try:
            violations = []
            
            # Step 1: Extract factual claims from the response.
            claims = self._extract_factual_claims(assistant_response)
            
            # Step 2: Check each claim for grounding in the context.
            grounding_violations = self._check_grounding(claims, context)
            violations.extend(grounding_violations)

            # Step 3: Check for critical safety violations (e.g., medical directives).
            safety_violations = self._check_safety(assistant_response)
            violations.extend(safety_violations)

            if not violations:
                logger.info("Guard approved response.")
                return True, assistant_response, "Approved: Response is grounded and safe."
            else:
                reasoning = "; ".join([v['description'] for v in violations])
                logger.warning(f"Guard rejected response: {reasoning}")
                fallback = self._get_appropriate_fallback(violations)
                return False, fallback, reasoning

        except Exception as e:
            logger.error(f"Guard evaluation failed unexpectedly: {e}")
            return False, self.fallback_messages["default"], str(e)

    def _check_grounding(self, claims: List[str], context: str) -> List[Dict[str, Any]]:
        """Checks if claims are semantically grounded in the context."""
        violations = []
        if not context: # If there's no context, any factual claim is ungrounded.
            if claims:
                violations.append({
                    "type": "ungrounded_claim",
                    "description": "Response contains claims but was generated without context."
                })
            return violations

        for claim in claims:
            score = self._calculate_grounding_score(claim, context)
            if score < 0.7: # Grounding threshold
                violations.append({
                    "type": "ungrounded_claim",
                    "description": f"Ungrounded claim found (score: {score:.2f}): '{claim[:50]}...'"
                })
        return violations

    def _check_safety(self, response: str) -> List[Dict[str, Any]]:
        """Checks for specific safety violation patterns."""
        violations = []
        # Check for directive medical language that isn't a direct quote.
        directive_patterns = [r'\byou (should|must|need to)\b', r'\b(start|stop|increase|decrease) (taking|your dose)\b']
        for pattern in directive_patterns:
            if re.search(pattern, response, re.IGNORECASE) and "according to the documentation" not in response.lower():
                violations.append({"type": "medical_advice", "description": "Contains a medical directive."})
                break
        return violations

    def _extract_factual_claims(self, response: str) -> List[str]:
        """A simple claim extractor. Splits response into sentences and filters out conversational filler."""
        sentences = re.split(r'[.!?]+', response)
        claims = []
        non_claim_patterns = [r'^(hello|hi|hey|thanks|thank you)', r'^\s*(i\'m sorry|i don\'t|i can\'t|i couldn\'t find)', r'^\s*(would you like|do you have|can i help)']
        
        for s in sentences:
            s = s.strip()
            if not s: continue
            is_filler = any(re.search(p, s, re.IGNORECASE) for p in non_claim_patterns)
            if not is_filler and len(s.split()) > 3: # Simple heuristic: a "fact" has more than 3 words.
                claims.append(s)
        return claims

    def _calculate_grounding_score(self, statement: str, context: str) -> float:
        """Calculates the max semantic similarity score between a statement and sentences in the context."""
        if not self.embedding_model or not context: return 0.0
        try:
            statement_embedding = self.embedding_model.encode(statement)
            context_sentences = re.split(r'[.!?]+', context)
            context_embeddings = self.embedding_model.encode([s for s in context_sentences if s.strip()])
            
            if len(context_embeddings) == 0: return 0.0
            
            # Calculate cosine similarity
            sims = np.dot(context_embeddings, statement_embedding) / (np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(statement_embedding))
            return float(np.max(sims))
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            return 0.0
            
    def _get_appropriate_fallback(self, violations: List[Dict[str, Any]]) -> str:
        """Selects a fallback message based on the most severe violation."""
        if any(v['type'] == 'medical_advice' for v in violations):
            return self.fallback_messages["unsafe_medical"]
        if any(v['type'] == 'ungrounded_claim' for v in violations):
            return self.fallback_messages["ungrounded"]
        return self.fallback_messages["default"]

# Global instance
guard_agent = SimplifiedGuardAgent()

# Public API function for your main app to call
def evaluate_response(context: str, assistant_response: str) -> Tuple[bool, str, str]:
    return guard_agent.evaluate_response(context, assistant_response)