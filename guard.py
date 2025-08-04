import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import re
from llm_client import call_guard_agent
from prompt import format_guard_prompt
from config import DEFAULT_FALLBACK_MESSAGE, ENABLE_GUARD, EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGuardAgent:
    """
    Intent-aware guard agent that evaluates responses based on conversational context and safety requirements.
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_messages = {
            "default": "I can't help with that. Can we talk about something else?",
            "no_context": "I don't have sufficient information in our knowledge base to answer that. Could you rephrase your question or ask about something else?",
            "unsafe_medical": "I cannot provide medical advice. Please consult with a healthcare professional for medical guidance.",
            "no_info": "I'm sorry, I don't seem to have any information on that topic in our documentation. Can I help you with something else?"
        }
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = None
        self._init_embedding_model()
        
        # Response intent types
        self.intent_types = {
            "ANSWERING": "Providing requested information",
            "ACKNOWLEDGING_GAP": "Explaining lack of information",
            "CONVERSATIONAL_BRIDGE": "Social pleasantries or topic transitions", 
            "OFFERING_ALTERNATIVES": "Suggesting related documented topics",
            "CLARIFYING": "Asking for clarification or more details"
        }
        
        # Violation categories with severity levels
        self.violation_categories = {
            "ungrounded_claim": {
                "severity": "high",
                "description": "Factual claim not found in provided context"
            },
            "medical_advice": {
                "severity": "critical",
                "description": "Provides treatment recommendation beyond documentation"
            },
            "knowledge_leakage": {
                "severity": "high", 
                "description": "Uses information not in provided documents"
            },
            "off_label": {
                "severity": "critical",
                "description": "Discusses uses not in approved documentation"
            },
            "competitor_mention": {
                "severity": "medium",
                "description": "References competitor products"
            },
            "promotional_language": {
                "severity": "medium",
                "description": "Uses promotional or exaggerated claims"
            },
            "inappropriate_tone": {
                "severity": "low",
                "description": "Casual or dismissive tone for medical topics"
            }
        }
    
    def evaluate_response(
        self, 
        context: str, 
        user_question: str, 
        assistant_response: str,
        conversation_history: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        Evaluate response with intent recognition and contextual understanding.
        
        Args:
            context (str): The retrieved context from RAG
            user_question (str): The user's original question
            assistant_response (str): The assistant's draft response
            conversation_history (str, optional): Previous conversation context
            
        Returns:
            Tuple[bool, str, str]: (is_approved, final_response, guard_reasoning)
        """
        if not self.enabled:
            return True, assistant_response, "Guard disabled"
        
        try:
            # Step 1: Analyze response intent
            intent = self._analyze_response_intent(assistant_response, user_question)
            logger.info(f"Identified response intent: {intent}")
            
            # Step 2: Extract claims based on intent
            claims = self._extract_claims_by_intent(assistant_response, intent)
            
            # Step 3: Check violations based on intent
            violations = self._check_violations_by_intent(
                assistant_response, context, claims, intent
            )
            
            # Step 4: Make nuanced verdict
            verdict_result = self._make_verdict(
                violations, intent, assistant_response, context, conversation_history
            )
            
            if verdict_result["verdict"] == "APPROVE":
                logger.info(f"Guard approved response - Intent: {intent}")
                return True, assistant_response, verdict_result["reasoning"]
            
            else:  # REJECT
                logger.warning(f"Guard rejected response: {verdict_result['reasoning']}")
                fallback = self._get_appropriate_fallback(verdict_result["violations"])
                return False, fallback, verdict_result["reasoning"]
                
        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}")
            return False, self.fallback_messages["default"], f"Guard evaluation error: {str(e)}"
    
    def _analyze_response_intent(self, response: str, question: str) -> str:
        """Identify the primary intent of the response"""
        response_lower = response.lower()
        question_lower = question.lower()
        
        # Check for conversational bridge patterns
        bridge_patterns = [
            "hello", "hi ", "good morning", "good afternoon", "thank you",
            "you're welcome", "happy to help", "glad to assist", "my pleasure"
        ]
        if any(pattern in response_lower[:50] for pattern in bridge_patterns):
            return "CONVERSATIONAL_BRIDGE"
        
        # Check for acknowledgment of missing information
        gap_patterns = [
            "don't have information", "don't have specific", "no information available",
            "not in our documentation", "not in the provided", "unable to find",
            "don't see any mention", "doesn't appear to be documented"
        ]
        if any(pattern in response_lower for pattern in gap_patterns):
            return "ACKNOWLEDGING_GAP"
        
        # Check for offering alternatives
        alternative_patterns = [
            "i can help with", "i can provide information about",
            "would you like to know about", "i can share information on",
            "related topics include", "alternatively"
        ]
        if any(pattern in response_lower for pattern in alternative_patterns):
            if any(gap in response_lower for gap in gap_patterns):
                return "OFFERING_ALTERNATIVES"
        
        # Check for clarification requests
        clarification_patterns = [
            "could you clarify", "could you specify", "which aspect",
            "do you mean", "are you asking about"
        ]
        if any(pattern in response_lower for pattern in clarification_patterns):
            return "CLARIFYING"
        
        # Default to answering if none of the above
        return "ANSWERING"
    
    def _extract_claims_by_intent(self, response: str, intent: str) -> List[Dict[str, Any]]:
        """Extract claims based on identified intent"""
        
        # Don't extract claims for certain intents
        if intent in ["CONVERSATIONAL_BRIDGE", "CLARIFYING"]:
            return []
        
        # For gap acknowledgment, only extract if making claims about what IS available
        if intent == "ACKNOWLEDGING_GAP":
            return self._extract_alternative_claims(response)
        
        # For answering and offering alternatives, extract all factual claims
        return self._extract_factual_claims(response)
    
    def _extract_alternative_claims(self, response: str) -> List[Dict[str, Any]]:
        """Extract claims about what information IS available"""
        claims = []
        
        # Patterns that indicate claims about available information
        available_patterns = [
            r"i can (provide|share|help with) information about ([^.]+)",
            r"our documentation (includes|covers|contains) ([^.]+)",
            r"available information includes ([^.]+)",
            r"we have (information|documentation) on ([^.]+)"
        ]
        
        for pattern in available_patterns:
            matches = re.finditer(pattern, response.lower())
            for match in matches:
                claim_text = match.group(0)
                claims.append({
                    "text": claim_text,
                    "type": "available_info",
                    "requires_grounding": True
                })
        
        return claims
    
    def _check_violations_by_intent(
        self, 
        response: str, 
        context: str, 
        claims: List[Dict[str, Any]], 
        intent: str
    ) -> List[Dict[str, Any]]:
        """Check for violations based on intent"""
        violations = []
        
        # Intent-specific violation checking
        if intent == "CONVERSATIONAL_BRIDGE":
            # Very lenient - only check for egregious issues
            if self._contains_medical_directive(response):
                violations.append({
                    "type": "medical_advice",
                    "severity": "critical",
                    "description": "Medical directive in conversational response"
                })
        
        elif intent == "ACKNOWLEDGING_GAP":
            # Check that it doesn't make ungrounded claims while acknowledging gaps
            for claim in claims:
                if claim.get("requires_grounding") and not self._is_claim_grounded(claim["text"], context):
                    violations.append({
                        "type": "ungrounded_claim",
                        "severity": "high",
                        "description": f"Claims availability of information not in context: {claim['text']}"
                    })
        
        elif intent in ["ANSWERING", "OFFERING_ALTERNATIVES"]:
            # Full validation for factual responses
            violations.extend(self._validate_factual_response(response, context, claims))
        
        return violations
    
    def _validate_factual_response(
        self, 
        response: str, 
        context: str, 
        claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Comprehensive validation for factual responses"""
        violations = []
        
        # Check each claim for grounding
        for claim in claims:
            grounding_score = self._calculate_grounding_score(claim["text"], context)
            
            if grounding_score < 0.7:
                violations.append({
                    "type": "ungrounded_claim",
                    "severity": "high",
                    "claim": claim["text"],
                    "score": grounding_score,
                    "description": f"Insufficient grounding (score: {grounding_score:.2f})"
                })
        
        # Check for medical safety violations
        medical_violations = self._check_medical_safety(response, context)
        violations.extend(medical_violations)
        
        # Check for promotional language
        if self._contains_promotional_language(response):
            violations.append({
                "type": "promotional_language",
                "severity": "medium",
                "description": "Contains promotional or exaggerated claims"
            })
        
        return violations
    
    def _make_verdict(
        self, 
        violations: List[Dict[str, Any]], 
        intent: str, 
        response: str,
        context: str,
        conversation_history: Optional[str]
    ) -> Dict[str, Any]:
        """Make binary verdict based on violations and intent"""
        
        # Any violation results in rejection for maximum safety
        if violations:
            # Get the most severe violation for reporting
            severity_order = ["critical", "high", "medium", "low"]
            violations_by_severity = {}
            for violation in violations:
                severity = violation.get("severity", "low")
                if severity not in violations_by_severity:
                    violations_by_severity[severity] = []
                violations_by_severity[severity].append(violation)
            
            # Find the most severe violation type
            most_severe = None
            for severity in severity_order:
                if severity in violations_by_severity and violations_by_severity[severity]:
                    most_severe = violations_by_severity[severity][0]
                    break
            
            return {
                "verdict": "REJECT",
                "reasoning": f"Safety violation detected: {most_severe['description']}",
                "violations": violations,
                "intent": intent
            }
        
        # No violations - approve
        return {
            "verdict": "APPROVE",
            "reasoning": f"Response approved - Intent: {intent}",
            "intent": intent
        }
    
    def _contains_medical_directive(self, response: str) -> bool:
        """Check for directive medical language"""
        directive_patterns = [
            r'\byou (should|must|need to) take\b',
            r'\bstart (taking|with)\b.*\bmg\b',
            r'\bstop taking\b',
            r'\bincrease your dose\b',
            r'\bdecrease your dose\b',
            r'\btake \d+\s*mg\b',
            r'\bdo not take\b.*\bif you\b'
        ]
        
        response_lower = response.lower()
        for pattern in directive_patterns:
            if re.search(pattern, response_lower):
                # Check if it's quoting context
                quote_indicators = ["according to", "documentation states", "guidelines say"]
                if not any(indicator in response_lower for indicator in quote_indicators):
                    return True
        
        return False
    
    def _check_medical_safety(self, response: str, context: str) -> List[Dict[str, Any]]:
        """Check for medical safety violations"""
        violations = []
        response_lower = response.lower()
        
        # Check for unqualified medical advice
        advice_patterns = [
            (r'effective for treating', 'efficacy_claim'),
            (r'will help with', 'efficacy_claim'),
            (r'cures?', 'efficacy_claim'),
            (r'safe for everyone', 'safety_overstatement'),
            (r'no side effects', 'safety_overstatement'),
            (r'completely safe', 'safety_overstatement')
        ]
        
        for pattern, violation_type in advice_patterns:
            if re.search(pattern, response_lower):
                # Check if claim is in context
                if not self._is_claim_grounded(pattern, context):
                    violations.append({
                        "type": "medical_advice",
                        "severity": "high",
                        "description": f"Ungrounded {violation_type.replace('_', ' ')}"
                    })
        
        return violations
    
    def _contains_promotional_language(self, response: str) -> bool:
        """Check for promotional or exaggerated language"""
        promotional_patterns = [
            r'\bbest\b.*\b(drug|medication|treatment)\b',
            r'\bmost effective\b',
            r'\bguaranteed\b',
            r'\bamazing results\b',
            r'\bbreakthrough\b',
            r'\bmiraculous\b',
            r'\brevolutionary\b'
        ]
        
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in promotional_patterns)
    
    def _is_claim_grounded(self, claim: str, context: str) -> bool:
        """Quick check if a claim appears to be grounded in context"""
        if not context:
            return False
        
        # Simple keyword overlap check
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        claim_words = claim_words - stop_words
        context_words = context_words - stop_words
        
        if not claim_words:
            return True  # Empty claim is "grounded"
        
        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap > 0.5
    
    def _get_appropriate_fallback(self, violations: List[Dict[str, Any]]) -> str:
        """Select appropriate fallback message based on violations"""
        
        # Check violation types
        violation_types = [v.get("type") for v in violations]
        
        if "medical_advice" in violation_types:
            return self.fallback_messages["unsafe_medical"]
        elif "ungrounded_claim" in violation_types:
            return self.fallback_messages["no_context"]
        else:
            return self.fallback_messages["default"]
    
    def _generate_coaching_feedback(
        self, 
        violations: List[Dict[str, Any]], 
        intent: str
    ) -> str:
        """Generate coaching feedback for rewrite suggestions"""
        feedback_parts = []
        
        if intent == "ANSWERING":
            feedback_parts.append("When providing information:")
        
        for violation in violations:
            if violation["type"] == "ungrounded_claim":
                feedback_parts.append("- Ensure all claims are directly supported by the provided context")
            elif violation["type"] == "medical_advice":
                feedback_parts.append("- Avoid directive language; quote documentation instead")
            elif violation["type"] == "promotional_language":
                feedback_parts.append("- Use neutral, factual language without superlatives")
        
        return "\n".join(feedback_parts)
    
    def _init_embedding_model(self):
        """Initialize the embedding model for semantic similarity"""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Initialized embedding model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def _extract_factual_claims(self, response: str) -> List[Dict[str, Any]]:
        """Extract factual claims from response"""
        import re
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        
        claims = []
        
        # Patterns indicating factual claims
        factual_indicators = [
            r'\b(?:is|are|was|were)\s+\w+',
            r'\b(?:contains?|includes?|has|have)\s+\w+',
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+',
            r'\b(?:works?\s+by|functions?\s+through)\s+\w+',
            r'\b\d+\s*(?:percent|%|mg|ml|mcg|iu|units?)\b',
            r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',
        ]
        
        # Exclude conversational patterns
        conversational_patterns = [
            r'^(?:hello|hi|hey|thanks|thank you)',
            r'^\s*(?:i\'m sorry|i don\'t|i can\'t)',
            r'^\s*(?:would you like|do you have)',
            r'^\s*(?:based on|according to)'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Check if conversational
            is_conversational = any(
                re.search(pattern, sentence.lower()) 
                for pattern in conversational_patterns
            )
            if is_conversational:
                continue
            
            # Check if contains factual indicators
            contains_factual = any(
                re.search(pattern, sentence.lower()) 
                for pattern in factual_indicators
            )
            
            if contains_factual:
                claims.append({
                    "text": sentence,
                    "type": "factual",
                    "requires_grounding": True
                })
        
        return claims
    
    def _calculate_grounding_score(self, statement: str, context: str) -> float:
        """Calculate semantic similarity between statement and context"""
        if not self.embedding_model or not context:
            return 0.0
        
        try:
            # Encode statement
            statement_embedding = self.embedding_model.encode(statement, convert_to_tensor=False, show_progress_bar=False)
            
            # Split context into sentences
            context_sentences = re.split(r'[.!?]+', context)
            context_sentences = [s.strip() for s in context_sentences if len(s.strip()) > 20]
            
            if not context_sentences:
                context_embedding = self.embedding_model.encode(context, convert_to_tensor=False, show_progress_bar=False)
                return float(np.dot(statement_embedding, context_embedding) / 
                           (np.linalg.norm(statement_embedding) * np.linalg.norm(context_embedding)))
            
            # Find best matching sentence
            max_similarity = 0.0
            for ctx_sentence in context_sentences:
                ctx_embedding = self.embedding_model.encode(ctx_sentence, convert_to_tensor=False, show_progress_bar=False)
                
                similarity = float(np.dot(statement_embedding, ctx_embedding) / 
                                 (np.linalg.norm(statement_embedding) * np.linalg.norm(ctx_embedding)))
                
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            return 0.0


# Global guard agent instance
guard_agent = EnhancedGuardAgent()

# Public API functions
def evaluate_response(
    context: str, 
    user_question: str, 
    assistant_response: str,
    conversation_history: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Evaluate assistant response with enhanced intent-based analysis.
    
    Args:
        context (str): The retrieved context from RAG
        user_question (str): The user's original question  
        assistant_response (str): The assistant's draft response
        conversation_history (str, optional): Previous conversation context
        
    Returns:
        Tuple[bool, str, str]: (is_approved, final_response, guard_reasoning)
    """
    return guard_agent.evaluate_response(context, user_question, assistant_response, conversation_history)

# Legacy functions for backward compatibility
def is_safe_response(text: str) -> bool:
    """Legacy function - returns basic safety check"""
    return len(text.strip()) >= 10

def redirect_response() -> str:
    """Legacy function - returns fallback message"""
    return DEFAULT_FALLBACK_MESSAGE