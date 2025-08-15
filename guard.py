# guard.py
import logging
from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np
from prompt import format_guard_prompt
from config import DEFAULT_FALLBACK_MESSAGE, ENABLE_GUARD, EMBEDDING_MODEL_NAME
from embeddings import get_embedding_model

logger = logging.getLogger(__name__)

class EnhancedGuardAgent:
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_messages = {
            "default": "I can't help with that. Can we talk about something else?",
            "no_context": "I don't have sufficient information in our knowledge base to answer that. Could you rephrase your question or ask about something else?",
            "unsafe_medical": "I cannot provide medical advice. Please consult with a healthcare professional for medical guidance.",
            "no_info": "I'm sorry, I don't seem to have any information on that topic in our documentation. Can I help you with something else?"
        }
        # Shared embedder
        self.embedding_model = get_embedding_model()

        self.intent_types = {
            "ANSWERING": "Providing requested information",
            "ACKNOWLEDGING_GAP": "Explaining lack of information",
            "CONVERSATIONAL_BRIDGE": "Social pleasantries or topic transitions",
            "OFFERING_ALTERNATIVES": "Suggesting related documented topics",
            "CLARIFYING": "Asking for clarification or more details"
        }

        self.violation_categories = {
            "ungrounded_claim": {"severity": "high", "description": "Factual claim not found in provided context"},
            "medical_advice": {"severity": "critical", "description": "Provides treatment recommendation beyond documentation"},
            "knowledge_leakage": {"severity": "high", "description": "Uses information not in provided documents"},
            "off_label": {"severity": "critical", "description": "Discusses uses not in approved documentation"},
            "competitor_mention": {"severity": "medium", "description": "References competitor products"},
            "promotional_language": {"severity": "medium", "description": "Uses promotional or exaggerated claims"},
            "inappropriate_tone": {"severity": "low", "description": "Casual or dismissive tone for medical topics"},
        }

    def evaluate_response(
        self,
        context: str,
        user_question: str,
        assistant_response: str,
        conversation_history: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        if not self.enabled:
            return True, assistant_response, "Guard disabled"
        try:
            intent = self._analyze_response_intent(assistant_response, user_question)
            logger.info(f"Identified response intent: {intent}")

            claims = self._extract_claims_by_intent(assistant_response, intent)
            violations = self._check_violations_by_intent(
                assistant_response, context, claims, intent
            )
            verdict_result = self._make_verdict(
                violations, intent, assistant_response, context, conversation_history
            )

            if verdict_result["verdict"] == "APPROVE":
                return True, assistant_response, verdict_result["reasoning"]
            else:
                fallback = self._get_appropriate_fallback(verdict_result["violations"])
                return False, fallback, verdict_result["reasoning"]

        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}", exc_info=True)
            return False, self.fallback_messages["default"], f"Guard evaluation error: {str(e)}"

    def _analyze_response_intent(self, response: str, question: str) -> str:
        rl = response.lower()
        bridge_patterns = ["hello", "hi ", "good morning", "good afternoon", "thank you",
                           "you're welcome", "happy to help", "glad to assist", "my pleasure"]
        if any(p in rl[:50] for p in bridge_patterns):
            return "CONVERSATIONAL_BRIDGE"

        gap_patterns = ["don't have information", "don't have specific", "no information available",
                        "not in our documentation", "not in the provided", "unable to find",
                        "don't see any mention", "doesn't appear to be documented"]
        if any(p in rl for p in gap_patterns):
            return "ACKNOWLEDGING_GAP"

        alternative_patterns = ["i can help with", "i can provide information about",
                                "would you like to know about", "i can share information on",
                                "related topics include", "alternatively"]
        if any(p in rl for p in alternative_patterns) and any(g in rl for g in gap_patterns):
            return "OFFERING_ALTERNATIVES"

        clarification_patterns = ["could you clarify", "could you specify", "which aspect",
                                  "do you mean", "are you asking about"]
        if any(p in rl for p in clarification_patterns):
            return "CLARIFYING"

        return "ANSWERING"

    def _extract_claims_by_intent(self, response: str, intent: str) -> List[Dict[str, Any]]:
        if intent in ["CONVERSATIONAL_BRIDGE", "CLARIFYING"]:
            return []
        if intent == "ACKNOWLEDGING_GAP":
            return self._extract_alternative_claims(response)
        return self._extract_factual_claims(response)

    def _extract_alternative_claims(self, response: str) -> List[Dict[str, Any]]:
        claims = []
        patterns = [
            r"i can (provide|share|help with) information about ([^.]+)",
            r"our documentation (includes|covers|contains) ([^.]+)",
            r"available information includes ([^.]+)",
            r"we have (information|documentation) on ([^.]+)"
        ]
        rl = response.lower()
        for pattern in patterns:
            for m in re.finditer(pattern, rl):
                claims.append({"text": m.group(0), "type": "available_info", "requires_grounding": True})
        return claims

    def _check_violations_by_intent(
        self,
        response: str,
        context: str,
        claims: List[Dict[str, Any]],
        intent: str
    ) -> List[Dict[str, Any]]:
        violations = []

        # **HARD RULE**: if answering or suggesting and there is NO context, reject
        if intent in ["ANSWERING", "OFFERING_ALTERNATIVES"] and not (context and context.strip()):
            return [{
                "type": "ungrounded_claim",
                "severity": "high",
                "description": "Factual response attempted without retrieved context"
            }]

        if intent == "CONVERSATIONAL_BRIDGE":
            if self._contains_medical_directive(response):
                violations.append({"type": "medical_advice", "severity": "critical",
                                   "description": "Medical directive in conversational response"})

        elif intent == "ACKNOWLEDGING_GAP":
            for claim in claims:
                if claim.get("requires_grounding") and not self._is_claim_grounded(claim["text"], context):
                    violations.append({"type": "ungrounded_claim", "severity": "high",
                                       "description": f"Claims availability not in context: {claim['text']}"})

        elif intent in ["ANSWERING", "OFFERING_ALTERNATIVES"]:
            violations.extend(self._validate_factual_response(response, context, claims))

        elif intent == "CLARIFYING":
            if self._contains_medical_directive(response):
                violations.append({"type": "medical_advice", "severity": "critical",
                                   "description": "Medical directive in clarifying response"})

        return violations

    def _validate_factual_response(self, response: str, context: str, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        violations = []
        for claim in claims:
            score = self._calculate_grounding_score(claim["text"], context)
            if score < 0.7:
                violations.append({
                    "type": "ungrounded_claim",
                    "severity": "high",
                    "claim": claim["text"],
                    "score": score,
                    "description": f"Insufficient grounding (score: {score:.2f})"
                })

        medical_violations = self._check_medical_safety(response, context)
        violations.extend(medical_violations)

        if self._contains_promotional_language(response):
            violations.append({"type": "promotional_language", "severity": "medium",
                               "description": "Contains promotional or exaggerated claims"})
        return violations

    def _make_verdict(self, violations: List[Dict[str, Any]], intent: str, response: str,
                      context: str, conversation_history: Optional[str]) -> Dict[str, Any]:
        if violations:
            severity_order = ["critical", "high", "medium", "low"]
            by_sev = {}
            for v in violations:
                by_sev.setdefault(v.get("severity", "low"), []).append(v)
            most_severe = next((by_sev[s][0] for s in severity_order if s in by_sev and by_sev[s]), violations[0])
            return {"verdict": "REJECT", "reasoning": f"Safety violation detected: {most_severe['description']}",
                    "violations": violations, "intent": intent}
        return {"verdict": "APPROVE", "reasoning": f"Response approved - Intent: {intent}", "intent": intent}

    def _contains_medical_directive(self, response: str) -> bool:
        patterns = [
            r'\byou (should|must|need to) take\b', r'\bstart (taking|with)\b.*\bmg\b',
            r'\bstop taking\b', r'\bincrease your dose\b', r'\bdecrease your dose\b',
            r'\btake \d+\s*mg\b', r'\bdo not take\b.*\bif you\b'
        ]
        rl = response.lower()
        for p in patterns:
            if re.search(p, rl):
                if not any(ind in rl for ind in ["according to", "documentation states", "guidelines say"]):
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
            (r'completely safe', 'safety_overstatement')
        ]
        for pattern, vtype in checks:
            if re.search(pattern, rl):
                if not self._is_claim_grounded(pattern, context):
                    violations.append({"type": "medical_advice", "severity": "high",
                                       "description": f"Ungrounded {vtype.replace('_', ' ')}"})
        return violations

    def _contains_promotional_language(self, response: str) -> bool:
        patterns = [
            r'\bbest\b.*\b(drug|medication|treatment)\b', r'\bmost effective\b', r'\bguaranteed\b',
            r'\bamazing results\b', r'\bbreakthrough\b', r'\bmiraculous\b', r'\brevolutionary\b'
        ]
        rl = response.lower()
        return any(re.search(p, rl) for p in patterns)

    def _is_claim_grounded(self, claim: str, context: str) -> bool:
        if not context:
            return False
        claim_words = set(re.findall(r'\b\w+\b', claim.lower())) - {'the','a','an','and','or','but','in','on','at','to','for'}
        context_words = set(re.findall(r'\b\w+\b', context.lower())) - {'the','a','an','and','or','but','in','on','at','to','for'}
        if not claim_words:
            return True
        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap > 0.5

    def _extract_factual_claims(self, response: str) -> List[Dict[str, Any]]:
        sentences = re.split(r'[.!?]+', response)
        claims = []
        indicators = [
            r'\b(?:is|are|was|were)\s+\w+', r'\b(?:contains?|includes?|has|have)\s+\w+',
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+', r'\b(?:works?\s+by|functions?\s+through)\s+\w+',
            r'\b\d+\s*(?:percent|%|mg|ml|mcg|iu|units?)\b', r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',
        ]
        conversational = [
            r'^(?:hello|hi|hey|thanks|thank you)', r'^\s*(?:i\'m sorry|i don\'t|i can\'t)',
            r'^\s*(?:would you like|do you have)', r'^\s*(?:based on|according to)',
            r'don\'t have (?:specific |any )?information', r'(?:not |don\'t see any )(?:mention|information|details)',
            r'unable to (?:find|provide)', r'(?:not |doesn\'t appear to be )(?:in|documented)',
            r'no (?:specific |documented )?(?:information|details|data)', r'knowledge base (?:doesn\'t|does not) (?:contain|include|have)'
        ]
        for s in sentences:
            s = s.strip()
            if len(s) < 10: continue
            if any(re.search(p, s.lower()) for p in conversational):
                continue
            if any(re.search(p, s.lower()) for p in indicators):
                claims.append({"text": s, "type": "factual", "requires_grounding": True})
        return claims

    def _calculate_grounding_score(self, statement: str, context: str) -> float:
        if not self.embedding_model or not context:
            return 0.0
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
            return 0.0

guard_agent = EnhancedGuardAgent()

def evaluate_response(context: str, user_question: str, assistant_response: str,
                      conversation_history: Optional[str] = None) -> Tuple[bool, str, str]:
    return guard_agent.evaluate_response(context, user_question, assistant_response, conversation_history)

def is_safe_response(text: str) -> bool:
    return len(text.strip()) >= 10

def redirect_response() -> str:
    return DEFAULT_FALLBACK_MESSAGE
