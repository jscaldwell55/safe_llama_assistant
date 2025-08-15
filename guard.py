# guard.py

import logging
from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np
import asyncio
import concurrent.futures
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
      3) LLM-based comprehensive safety evaluation (optional)
    """

    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.use_llm_guard = None  # Will be loaded from config
        self.embedding_model = _load_embedding_model()
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

    async def _call_guard_llm(self, prompt: str) -> str:
        """Call the guard LLM asynchronously"""
        from llm_client import call_guard_agent
        return await call_guard_agent(prompt)

    def _is_conversational_only(self, question: str, response: str) -> bool:
        """
        Determine if this is conversational exchange without pharmaceutical facts.
        Be PERMISSIVE - only return False if there's clear medical/drug content.
        """
        r_lower = response.lower()
        
        # Strong indicators this IS pharmaceutical/medical content (needs grounding)
        pharmaceutical_indicators = [
            r'\b\d+\s*(?:mg|ml|mcg|iu|units?)\b',  # Dosages
            r'\b(?:lexapro|escitalopram|ssri|antidepressant)\b',  # Drug names
            r'\b(?:dose|dosage|dosing|administration)\b',
            r'\b(?:side effect|adverse|reaction|interaction)\b',
            r'\b(?:indication|contraindication|warning)\b',
            r'\b(?:treatment|therapy|medication|prescription)\b',
            r'\b(?:clinical|efficacy|safety profile)\b',
            r'\b(?:take|taking|taken)\s+(?:with|without|before|after)\b',
            r'\b(?:oral|injection|intravenous|topical)\b',
            r'\b(?:overdose|withdrawal|dependence)\b',
        ]
        
        # If response contains pharmaceutical content, it needs grounding
        for pattern in pharmaceutical_indicators:
            if re.search(pattern, r_lower):
                return False
        
        # Otherwise, treat it as conversational
        # This allows the LLM to:
        # - Answer "how are you?" naturally
        # - Explain what it can help with
        # - Handle clarifications
        # - Provide encouragement
        # - Make appropriate referrals
        # - Be empathetic
        # - And much more...
        return True

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
        
        # Load LLM guard setting if not already loaded
        if self.use_llm_guard is None:
            from config import USE_LLM_GUARD
            self.use_llm_guard = USE_LLM_GUARD
        
        try:
            # Check if this is pure conversation (no medical content)
            if self._is_conversational_only(user_question, assistant_response):
                return True, assistant_response, "Approve: Conversational response, no medical content"
            
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
            
            # Phase 3: LLM-based comprehensive evaluation (skip if disabled or for simple responses)
            if not self.use_llm_guard:
                # Just use heuristics and grounding
                llm_verdict = {
                    "verdict": "APPROVE" if grounding_score > self.sim_threshold else "UNCERTAIN",
                    "intent": "ANSWERING",
                    "reason": "LLM guard disabled, using heuristics only",
                    "confidence": 0.7
                }
            elif self._should_skip_llm_evaluation(assistant_response, grounding_score, claims):
                llm_verdict = {
                    "verdict": "APPROVE",
                    "intent": "ANSWERING",
                    "reason": "Simple response with good grounding, skipped LLM check",
                    "confidence": 0.85
                }
            else:
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

    def _should_skip_llm_evaluation(self, response: str, grounding_score: float, claims: List[str]) -> bool:
        """Determine if we can skip LLM evaluation for performance"""
        response_lower = response.lower()
        
        # Skip for excellent grounding regardless of content type
        if grounding_score > 0.75:
            # Check for obviously dangerous content even with good grounding
            critical_terms = ["inject", "snort", "crush", "overdose", "suicide", "kill"]
            if not any(term in response_lower for term in critical_terms):
                return True
        
        # Skip for short, simple responses
        if len(response) < 200 and grounding_score > 0.65:
            return True
        
        # Skip for list-based responses (bullets or numbered lists)
        if (response.count('•') > 2 or response.count('-') > 3 or 
            response.count('\n') > 4 or bool(re.search(r'\n\d+\.', response))):
            # List responses are usually just facts from documentation
            if grounding_score > 0.6:
                return True
        
        # Skip for responses that are mostly side effects or symptoms listings
        medical_list_indicators = ["side effects", "symptoms", "reactions", "effects include", 
                                  "common effects", "serious effects", "may include", "such as"]
        if any(indicator in response_lower for indicator in medical_list_indicators):
            if grounding_score > 0.65:
                return True
        
        # Skip for standard acknowledgment responses
        no_info_phrases = ["don't have information", "not in the documentation", 
                          "no information", "unable to find", "not available",
                          "cannot provide", "don't have specific"]
        if any(phrase in response_lower for phrase in no_info_phrases):
            return True
        
        # Skip for conversational responses with high grounding and no claims
        if grounding_score > 0.7 and not claims:
            return True
        
        # Skip for factual responses without directive language
        if grounding_score > 0.7 and claims:
            risky_phrases = ["you should", "you must", "you need to", "recommend that you",
                           "advise you", "suggest you", "do not take", "never take", 
                           "always take", "stop taking", "start taking", "increase your",
                           "decrease your", "double your", "skip your"]
            if not any(phrase in response_lower for phrase in risky_phrases):
                return True
        
        # Skip for definition/explanation responses
        definition_patterns = [r"^.{0,50}\bis\b.{0,200}$", r"refers to", r"means that", 
                             r"is defined as", r"is a type of", r"is used for"]
        if any(re.search(pattern, response_lower) for pattern in definition_patterns):
            if grounding_score > 0.65 and len(response) < 300:
                return True
        
        # Skip for responses that just quote numbers/statistics
        if re.search(r'\d+\s*(?:%|percent|mg|ml|mcg|patients|people|studies)', response_lower):
            if grounding_score > 0.65 and "should" not in response_lower and "must" not in response_lower:
                return True
        
        # Default to running LLM evaluation for anything else
        return False

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

    def _run_async_safe(self, coro):
        """Run an async coroutine safely, handling existing event loops"""
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, we can use asyncio.run()
            return asyncio.run(coro)
        
        # There's a running loop (likely from Streamlit), use thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    # ---------- Phase 3: LLM evaluation ----------
    def _run_llm_evaluation(
        self, context: str, question: str, response: str, 
        conversation_history: Optional[str], grounding_score: float
    ) -> Dict[str, Any]:
        """Run LLM-based evaluation (with proper async handling)"""
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
            
            # Call the LLM with proper async handling
            llm_response = self._run_async_safe(self._call_guard_llm(prompt))
            
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
        """
        Extract only pharmaceutical/medical factual claims that need grounding.
        Be permissive - conversational content doesn't count as 'claims'.
        """
        if not response:
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        sentences = [s.strip() for s in sentences if s and len(s.strip()) > 3]

        claims: List[str] = []
        
        # Only look for sentences with clear medical/pharmaceutical facts
        pharmaceutical_claim_indicators = [
            r'\b\d+\s*(?:mg|ml|mcg|iu|units?)\b',  # Dosages
            r'\b(?:lexapro|escitalopram|ssri)\b',  # Specific drugs
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+',  # Causal medical claims
            r'\b(?:treats?|treatment|therapy|medication)\b',
            r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',
            r'\b(?:side effect|adverse|interaction)\b',
            r'\b(?:clinical|efficacy|safety)\b',
        ]
        
        for s in sentences:
            sl = s.lower()
            # Only count as a claim if it contains pharmaceutical/medical facts
            if any(re.search(p, sl) for p in pharmaceutical_claim_indicators):
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