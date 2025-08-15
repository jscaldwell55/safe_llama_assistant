# guard.py

import logging
from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np
import asyncio
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

# --- Patterns for critical checks (kept for pre-filtering obvious violations) ---
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
DOSING_DIRECTIVES = [
    r'\byou (should|must|need to) take\b', r'\bstart (taking|with)\b.*\bmg\b',
    r'\bstop taking\b', r'\bincrease your dose\b', r'\bdecrease your dose\b',
    r'\btake \d+\s*mg\b', r'\bdo not take\b.*\bif you\b'
]

class HybridGuardAgent:
    """
    Hybrid guard using both heuristics and LLM reasoning:
      1) Quick heuristic pre-checks for obvious violations
      2) Semantic grounding checks
      3) LLM-based comprehensive safety evaluation
    """

    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.embedding_model = _load_embedding_model()
        self.llm_client = None  # Will be lazy-loaded
        try:
            self.sim_threshold = float(SEMANTIC_SIMILARITY_THRESHOLD)
        except Exception:
            self.sim_threshold = 0.65

        # User-facing fallbacks
        self.fallback_messages = {
            "default": "I can't help with that. Can we talk about something else?",
            "no_context": "I don't have sufficient information in our knowledge base to answer that. Could you rephrase your question or ask about something else?",
            "unsafe_medical": "I cannot provide medical advice. Please consult with a healthcare professional for medical guidance.",
            "off_label_refusal": "I can't provide information about unapproved or off-label uses. I can share what's documented in the approved labeling.",
            "misuse_refusal": (
                "I can't help with questions about misusing or altering medications. "
                "If you're in immediate danger, call local emergency services. "
                "For confidential support with substance use, in the U.S. you can call SAMHSA at 1-800-662-HELP (4357)."
            ),
        }

    def _get_llm_client(self):
        """Lazy load the LLM client to avoid circular imports"""
        if self.llm_client is None:
            from llm_client import call_guard_agent
            self.llm_client = call_guard_agent
        return self.llm_client

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
            # Phase 1: Quick heuristic pre-checks for obvious violations
            obvious_violation = self._obvious_violation_check(user_question, assistant_response)
            if obvious_violation:
                fallback = self._fallback_for_violation_type(obvious_violation["type"])
                return False, fallback, f"Reject (heuristic): {obvious_violation['description']}"

            # Phase 2: Semantic grounding check
            claims = self._extract_factual_claims(assistant_response)
            if claims and not (context and context.strip()):
                return False, self.fallback_messages["no_context"], "Reject: factual content with no retrieved context."
            
            grounding_score = self._calculate_grounding_score(assistant_response, context)
            
            # Phase 3: LLM-based comprehensive evaluation (async to sync bridge)
            llm_verdict = self._run_llm_evaluation(
                context, user_question, assistant_response, 
                conversation_history, grounding_score
            )
            
            # Phase 4: Combine signals for final decision
            return self._final_verdict(
                assistant_response, grounding_score, llm_verdict, claims, context
            )

        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}", exc_info=True)
            return False, self.fallback_messages["default"], f"Guard evaluation error: {str(e)}"

    # ---------- Phase 1: Obvious violation checks ----------
    def _obvious_violation_check(self, question: str, response: str) -> Optional[Dict[str, Any]]:
        """Quick pattern-based check for obvious violations"""
        ql = question.lower()
        rl = response.lower()
        
        # Check for misuse intent
        if self._contains_pattern(ql, MISUSE_PATTERNS) or self._contains_pattern(rl, MISUSE_PATTERNS):
            return {"type": "misuse", "description": "Misuse or unapproved route detected"}
        
        # Check for obvious directive medical language
        if self._contains_pattern(rl, DOSING_DIRECTIVES):
            return {"type": "medical_advice", "description": "Directive medical language detected"}
        
        return None

    # ---------- Phase 2: Grounding score calculation ----------
    def _calculate_grounding_score(self, response: str, context: str) -> float:
        """Calculate overall grounding score for the response"""
        if not context:
            return 0.0
        
        # Get semantic similarity for the full response
        full_score = self._semantic_similarity(response, context)
        
        # Also check individual claims
        claims = self._extract_factual_claims(response)
        if not claims:
            return full_score
        
        claim_scores = [self._semantic_similarity(claim, context) for claim in claims]
        avg_claim_score = sum(claim_scores) / len(claim_scores) if claim_scores else 0
        
        # Weight both full response and individual claims
        return 0.4 * full_score + 0.6 * avg_claim_score

    # ---------- Phase 3: LLM evaluation ----------
    def _run_llm_evaluation(
        self, context: str, question: str, response: str, 
        conversation_history: Optional[str], grounding_score: float
    ) -> Dict[str, Any]:
        """Run LLM-based evaluation (with async-to-sync bridge)"""
        try:
            # Format the prompt for the guard LLM
            from prompts import format_guard_prompt
            prompt = format_guard_prompt(
                context=context,
                question=question,
                answer=response,
                conversation_history=conversation_history
            )
            
            # Add grounding score to help LLM
            prompt += f"\n\nGrounding Score: {grounding_score:.2f}"
            
            # Call the LLM (handle async)
            llm_response = asyncio.run(self._get_llm_client()(prompt))
            
            # Parse LLM verdict
            return self._parse_llm_verdict(llm_response)
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}", exc_info=True)
            # Fallback to heuristics if LLM fails
            return {"verdict": "UNCERTAIN", "reason": f"LLM evaluation failed: {str(e)}"}

    def _parse_llm_verdict(self, llm_response: str) -> Dict[str, Any]:
        """Parse the LLM's verdict from its response"""
        response_lower = llm_response.lower()
        
        # Look for clear verdict signals
        if "approve" in response_lower and "reject" not in response_lower:
            # Extract intent and reason
            intent_match = re.search(r'\[Intent:\s*([^\]]+)\]', llm_response, re.IGNORECASE)
            reason_match = re.search(r'\[(?:Brief\s+)?reason:\s*([^\]]+)\]', llm_response, re.IGNORECASE)
            
            return {
                "verdict": "APPROVE",
                "intent": intent_match.group(1) if intent_match else "UNKNOWN",
                "reason": reason_match.group(1) if reason_match else llm_response,
                "confidence": self._extract_confidence(llm_response)
            }
        
        elif "reject" in response_lower:
            # Extract violation type and issue
            violation_match = re.search(r'\[([^\]]+)\]', llm_response)
            issue_match = re.search(r'\[Specific issue:\s*([^\]]+)\]', llm_response, re.IGNORECASE)
            
            return {
                "verdict": "REJECT",
                "violation_type": violation_match.group(1) if violation_match else "UNKNOWN",
                "issue": issue_match.group(1) if issue_match else llm_response,
                "confidence": self._extract_confidence(llm_response)
            }
        
        else:
            return {
                "verdict": "UNCERTAIN",
                "reason": llm_response,
                "confidence": 0.5
            }

    def _extract_confidence(self, llm_response: str) -> float:
        """Extract confidence score if mentioned by LLM"""
        confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', llm_response, re.IGNORECASE)
        if confidence_match:
            try:
                return float(confidence_match.group(1))
            except:
                pass
        # Default confidence based on verdict clarity
        if "strongly" in llm_response.lower() or "clearly" in llm_response.lower():
            return 0.9
        elif "uncertain" in llm_response.lower() or "unclear" in llm_response.lower():
            return 0.5
        return 0.75

    # ---------- Phase 4: Final verdict combination ----------
    def _final_verdict(
        self, 
        response: str,
        grounding_score: float,
        llm_verdict: Dict[str, Any],
        claims: List[str],
        context: str
    ) -> Tuple[bool, str, str]:
        """Combine all signals for final decision"""
        
        # Strong rejection from LLM with high confidence
        if llm_verdict["verdict"] == "REJECT" and llm_verdict.get("confidence", 0) > 0.7:
            violation_type = llm_verdict.get("violation_type", "safety_violation")
            fallback = self._fallback_for_violation_type(violation_type)
            return False, fallback, f"Reject (LLM): {llm_verdict.get('issue', 'Safety violation detected')}"
        
        # Poor grounding score (even if LLM approves)
        if grounding_score < self.sim_threshold and claims:
            # Double-check with lexical grounding as fallback
            if not self._lexically_grounded(response, context):
                return False, self.fallback_messages["no_context"], f"Reject: Poor grounding (score={grounding_score:.2f})"
        
        # LLM uncertain but grounding is good
        if llm_verdict["verdict"] == "UNCERTAIN" and grounding_score > self.sim_threshold:
            # Trust grounding and approve with caution
            return True, response, f"Approve with caution: Good grounding ({grounding_score:.2f}), LLM uncertain"
        
        # LLM approves with good grounding
        if llm_verdict["verdict"] == "APPROVE" and grounding_score > self.sim_threshold:
            intent = llm_verdict.get("intent", "ANSWERING")
            return True, response, f"Approve: LLM approved (intent={intent}), grounding={grounding_score:.2f}"
        
        # Default to safety when uncertain
        if llm_verdict["verdict"] == "UNCERTAIN":
            return False, self.fallback_messages["default"], "Reject: Uncertain safety evaluation"
        
        # Approve if LLM approves (even with lower grounding for conversational responses)
        if llm_verdict["verdict"] == "APPROVE":
            intent = llm_verdict.get("intent", "ANSWERING")
            return True, response, f"Approve: LLM approved (intent={intent})"
        
        # Default reject
        return False, self.fallback_messages["default"], "Reject: Failed comprehensive evaluation"

    # ---------- Helper methods (kept from original) ----------
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

    def _fallback_for_violation_type(self, violation_type: str) -> str:
        violation_lower = violation_type.lower()
        if "misuse" in violation_lower:
            return self.fallback_messages["misuse_refusal"]
        elif "off" in violation_lower and "label" in violation_lower:
            return self.fallback_messages["off_label_refusal"]
        elif "medical" in violation_lower or "dosage" in violation_lower:
            return self.fallback_messages["unsafe_medical"]
        elif "ground" in violation_lower:
            return self.fallback_messages["no_context"]
        else:
            return self.fallback_messages["default"]

# Global instance + wrapper
guard_agent = HybridGuardAgent()

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