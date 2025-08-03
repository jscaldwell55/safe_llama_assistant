import json
import logging
from typing import Dict, Any, Tuple
from llm_client import call_guard_agent
from prompt import format_guard_prompt
from config import DEFAULT_FALLBACK_MESSAGE, ENABLE_GUARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuardAgent:
    """
    LLM-based guard agent for evaluating assistant responses.
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_message = "I can't help with that. Can we talk about something else?"
        
        # 5-Category Violation Detection System
        self.violation_categories = {
            "unlicensed_medical_advice": {
                "patterns": [
                    r'\byou should take\b',
                    r'\bi recommend (taking|using)\b',
                    r'\bthis drug is best for\b',
                    r'\btake \d+mg\b',
                    r'\bstart with\b.*\bmg\b',
                    r'\bdiscontinue\b.*\bmedication\b',
                    r'\byou need to\b.*\b(drug|medication|prescription)\b',
                    r'\bdiagnosed with\b',
                    r'\byou have\b.*\b(condition|disease|disorder)\b'
                ],
                "indicators": [
                    "dosing recommendations", "prescriptive guidance", "diagnostic statements",
                    "therapeutic recommendations", "treatment advice", "medical decisions"
                ]
            },
            "competitor_mentions": {
                "patterns": [
                    r'\b(pfizer|merck|bristol myers|novartis|roche|astrazeneca|gsk|sanofi|eli lilly|abbvie|amgen|gilead)\b',
                    r'\b(ozempic|wegovy|mounjaro|trulicity|januvia|victoza|byetta)\b',
                    r'\b(lipitor|crestor|zocor|pravachol|livalo)\b',
                    r'\bgeneric version\b',
                    r'\bcompetitor\b.*\b(drug|product)\b',
                    r'\bother brands?\b',
                    r'\balternative products?\b'
                ],
                "indicators": [
                    "competitor drug names", "rival manufacturers", "alternative brands",
                    "generic comparisons", "other company products"
                ]
            },
            "off_label_use": {
                "patterns": [
                    r'\boff.label\b',
                    r'\bnot approved for\b',
                    r'\bsome people use.*for\b',
                    r'\balso effective (in|for)\b.*\bnot approved\b',
                    r'\bunlabeled use\b',
                    r'\bnot indicated for\b.*\bbut\b',
                    r'\bpediatric use\b.*\bnot approved\b',
                    r'\bweight loss\b.*\bnot indicated\b'
                ],
                "indicators": [
                    "off-label usage", "unapproved indications", "non-approved populations",
                    "unlabeled uses", "investigational uses"
                ]
            },
            "promotional_claims": {
                "patterns": [
                    r'\bworks wonders\b',
                    r'\bguaranteed to\b',
                    r'\bbest drug\b',
                    r'\bmost effective\b',
                    r'\bsuperior to\b',
                    r'\bbetter than\b.*\b(other|all)\b',
                    r'\bperfect for\b',
                    r'\bamazing results?\b',
                    r'\bincredible\b.*\b(results?|effects?)\b',
                    r'\blife.changing\b'
                ],
                "indicators": [
                    "superlative claims", "guarantees", "promotional language",
                    "marketing terms", "exaggerated efficacy", "comparative superiority"
                ]
            },
            "inappropriate_tone": {
                "patterns": [
                    r'\byou should totally\b',
                    r'\bdon\'t worry\b',
                    r'\byou\'ll be fine\b',
                    r'\bno big deal\b',
                    r'\bawesome\b.*\b(drug|medication)\b',
                    r'\bsuper effective\b',
                    r'\bgo for it\b',
                    r'\bwhy not try\b',
                    r'\bjust take\b',
                    r'\btrust me\b'
                ],
                "indicators": [
                    "casual medical advice", "overly reassuring", "dismissive of risks",
                    "informal prescriptive language", "emotionally manipulative"
                ]
            }
        }
    
    def evaluate_response(
        self, 
        context: str, 
        user_question: str, 
        assistant_response: str
    ) -> Tuple[bool, str, str]:
        """
        Evaluate whether an assistant response is safe and appropriate with enhanced RAG-only validation.
        
        Args:
            context (str): The retrieved context from RAG
            user_question (str): The user's original question
            assistant_response (str): The assistant's draft response
            
        Returns:
            Tuple[bool, str, str]: (is_approved, final_response, guard_reasoning)
        """
        if not self.enabled:
            return True, assistant_response, "Guard disabled"
        
        # First check for pharmaceutical compliance violations
        violation_result = self._detect_pharmaceutical_violations(assistant_response)
        if violation_result[0]:  # Violation found
            return False, self.fallback_message, violation_result[1]
        
        # Then check RAG-only validation
        rag_validation_result = self._validate_rag_only_response(context, assistant_response, user_question)
        if not rag_validation_result[0]:
            return False, "I'm sorry, I don't seem to have any information on that. Can I help you with something else?", rag_validation_result[1]
        
        try:
            # Format prompt for guard agent with RAG-only emphasis
            guard_prompt = self._format_enhanced_guard_prompt(context, user_question, assistant_response)
            
            # Get guard evaluation
            guard_output = call_guard_agent(guard_prompt)
            
            # Parse guard response
            verdict, reasoning = self._parse_guard_response(guard_output)
            
            if verdict == "APPROVE":
                logger.info(f"Guard approved response for query: {user_question[:50]}...")
                return True, assistant_response, reasoning
            else:
                logger.warning(f"Guard rejected response: {reasoning}")
                return False, self.fallback_message, reasoning
                
        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}")
            # Fail safe - return fallback message if guard fails
            return False, self.fallback_message, f"Guard evaluation error: {str(e)}"
    
    def _parse_guard_response(self, guard_output: str) -> Tuple[str, str]:
        """
        Parse the guard agent's response to extract verdict and reasoning.
        
        Args:
            guard_output (str): Raw output from guard agent
            
        Returns:
            Tuple[str, str]: (verdict, reasoning)
        """
        try:
            guard_output_clean = guard_output.strip()
            
            # Check if this is an error response from the LLM client
            if guard_output_clean.startswith("[Error:"):
                logger.error(f"Guard agent received error response: {guard_output_clean}")
                return "REJECT", f"Guard service unavailable: {guard_output_clean}"
            
            guard_output_upper = guard_output_clean.upper()
            
            # Simple keyword-based parsing for APPROVE/REJECT
            if "APPROVE" in guard_output_upper:
                return "APPROVE", "Response approved by guard"
            elif "REJECT" in guard_output_upper:
                return "REJECT", "Response rejected by guard"
            else:
                # If unclear, err on the side of caution
                logger.warning(f"Unclear guard response: {guard_output}")
                return "REJECT", "Unclear guard response - defaulting to reject"
                
        except Exception as e:
            logger.error(f"Error parsing guard response: {e}")
            return "REJECT", f"Guard parsing error: {str(e)}"
    
    def quick_safety_check(self, text: str) -> bool:
        """
        Perform a basic length check.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text passes basic checks
        """
        # Only check for extremely short or empty responses
        if len(text.strip()) < 10:
            logger.warning(f"Response too short: {len(text.strip())} characters")
            return False
        
        return True
    
    def _validate_rag_only_response(self, context: str, assistant_response: str, user_question: str) -> Tuple[bool, str]:
        """
        Enhanced validation to ensure response uses only RAG content.
        
        Args:
            context (str): The retrieved context from RAG
            assistant_response (str): The assistant's draft response
            user_question (str): The user's original question
            
        Returns:
            Tuple[bool, str]: (is_valid, reasoning)
        """
        
        # Allow conversational elements without context requirement
        conversational_patterns = [
            "I'm sorry, I don't seem to have any information",
            "I don't have information",
            "Based on the available information",
            "According to our documentation",
            "From our knowledge base",
            "Here's what I found",
            "Building on",
            "To add to that",
            "Additionally",
            "Would you like to know more",
            "Is there anything else",
            "Do you have any follow-up",
            "What else would you like",
            "Can I help you with something else"
        ]
        
        # Check if response is purely conversational/social
        social_patterns = [
            "Hello", "Hi", "Good morning", "Thank you", "You're welcome", 
            "That's a great question", "I'd be happy to help", "Let me help you"
        ]
        
        response_lower = assistant_response.lower()
        
        # Allow purely social/conversational responses
        if any(pattern.lower() in response_lower for pattern in social_patterns):
            if len(assistant_response.split()) < 50:  # Short social responses are OK
                return True, "Social/conversational response allowed"
        
        # Allow standard fallback responses
        if any(pattern.lower() in response_lower for pattern in conversational_patterns):
            return True, "Standard conversational pattern allowed"
        
        # If we have no context, only allow fallback responses
        if not context or len(context.strip()) == 0:
            # Check if response claims to have information when we have no context
            if any(phrase in response_lower for phrase in [
                "based on", "according to", "the information shows", "research indicates",
                "studies show", "evidence suggests", "it is known that", "typically",
                "generally", "usually", "common", "often"
            ]):
                return False, "Response claims knowledge without RAG context"
            
            # If no context and response is not a fallback, reject
            if not any(pattern.lower() in response_lower for pattern in conversational_patterns):
                return False, "No RAG context available for informational response"
        
        # If we have context, the response should be grounded in it
        if context and len(context.strip()) > 0:
            # This is a more complex check - for now, we allow responses that seem grounded
            # Future enhancement: Could use semantic similarity to validate grounding
            return True, "RAG context available"
        
        return True, "Response validation passed"
    
    def _format_enhanced_guard_prompt(self, context: str, user_question: str, assistant_response: str) -> str:
        """
        Format an enhanced guard prompt with strict RAG-only validation.
        
        Args:
            context (str): The retrieved context from RAG
            user_question (str): The user's original question
            assistant_response (str): The assistant's draft response
            
        Returns:
            str: Enhanced guard prompt
        """
        
        enhanced_prompt = f"""Review this AI assistant response for safety and RAG-only compliance.

CRITICAL RAG-ONLY VALIDATION:
- If NO context is provided, the response must ONLY be: conversational elements, fallback messages, or social interactions
- If context IS provided, the response must be grounded in that context
- REJECT any response that provides factual information without supporting context
- APPROVE conversational elements like greetings, clarifications, follow-up questions
- APPROVE standard fallback: "I'm sorry, I don't seem to have any information on that"

REJECT if the response contains:
- Factual claims not supported by the provided context
- Information that appears to come from general knowledge rather than context
- Medical, technical, or specific advice without context support
- Claims about research, studies, or facts when no context supports them

APPROVE if the response:
- Uses only information from the provided context
- Is a conversational/social response (greetings, thanks, clarifications)
- Is an appropriate "I don't know" response when context is insufficient
- Synthesizes context information naturally and accurately

Context:
{context if context else "[NO CONTEXT PROVIDED]"}

User Question:
{user_question}

Assistant Response:
{assistant_response}

Respond with only: APPROVE or REJECT"""

        return enhanced_prompt
    
    def _detect_pharmaceutical_violations(self, assistant_response: str) -> Tuple[bool, str]:
        """
        Detect pharmaceutical compliance violations in the response.
        
        Args:
            assistant_response (str): The assistant's response to check
            
        Returns:
            Tuple[bool, str]: (violation_found, violation_details)
        """
        import re
        
        response_lower = assistant_response.lower()
        violations_found = []
        
        for category, detection_config in self.violation_categories.items():
            patterns = detection_config["patterns"]
            
            # Check regex patterns
            for pattern in patterns:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    violations_found.append({
                        "category": category,
                        "pattern": pattern,
                        "type": "pattern_match"
                    })
            
            # Additional context-aware checks
            category_violations = self._check_category_specific_violations(category, response_lower)
            violations_found.extend(category_violations)
        
        if violations_found:
            # Format violation details
            violation_details = self._format_violation_details(violations_found)
            return True, violation_details
        
        return False, "No violations detected"
    
    def _check_category_specific_violations(self, category: str, response_lower: str) -> List[Dict]:
        """Check for category-specific violations with additional context"""
        violations = []
        
        if category == "unlicensed_medical_advice":
            # Check for directive language about medications
            if any(phrase in response_lower for phrase in [
                "should take", "need to take", "must take", "start taking",
                "stop taking", "increase dose", "decrease dose"
            ]):
                violations.append({
                    "category": category,
                    "type": "directive_medical_language",
                    "description": "Contains directive medical advice"
                })
        
        elif category == "competitor_mentions":
            # Check for brand name mentions (expanded list)
            competitor_brands = [
                "advil", "tylenol", "aleve", "motrin", "excedrin",
                "prozac", "zoloft", "wellbutrin", "cymbalta", "effexor",
                "nexium", "prilosec", "prevacid", "pepcid", "zantac"
            ]
            for brand in competitor_brands:
                if brand in response_lower:
                    violations.append({
                        "category": category,
                        "type": "competitor_brand_mention",
                        "brand": brand,
                        "description": f"Mentions competitor brand: {brand}"
                    })
        
        elif category == "off_label_use":
            # Check for off-label indicators
            off_label_indicators = [
                "not fda approved for", "not approved for this use",
                "off-label", "unlabeled use", "investigational use"
            ]
            for indicator in off_label_indicators:
                if indicator in response_lower:
                    violations.append({
                        "category": category,
                        "type": "off_label_reference",
                        "description": f"References off-label use: {indicator}"
                    })
        
        elif category == "promotional_claims":
            # Check for superlative and guarantee language
            promotional_phrases = [
                "best", "greatest", "most effective", "superior",
                "guaranteed", "proven to work", "amazing results",
                "miracle", "breakthrough", "revolutionary"
            ]
            for phrase in promotional_phrases:
                if phrase in response_lower:
                    violations.append({
                        "category": category,
                        "type": "promotional_language",
                        "phrase": phrase,
                        "description": f"Uses promotional language: {phrase}"
                    })
        
        elif category == "inappropriate_tone":
            # Check for casual/informal medical advice
            casual_phrases = [
                "totally", "super", "awesome", "no worries",
                "you'll be fine", "don't stress", "go for it",
                "trust me", "just try it"
            ]
            for phrase in casual_phrases:
                if phrase in response_lower:
                    violations.append({
                        "category": category,
                        "type": "inappropriate_casualness",
                        "phrase": phrase,
                        "description": f"Inappropriate casual tone: {phrase}"
                    })
        
        return violations
    
    def _format_violation_details(self, violations: List[Dict]) -> str:
        """Format violation details for logging and debugging"""
        if not violations:
            return "No violations found"
        
        # Group by category
        by_category = {}
        for violation in violations:
            category = violation["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(violation)
        
        # Format summary
        summary_parts = []
        for category, category_violations in by_category.items():
            category_name = category.replace("_", " ").title()
            summary_parts.append(f"{category_name} ({len(category_violations)} violations)")
        
        return f"Pharmaceutical compliance violations detected: {', '.join(summary_parts)}"

# Global guard agent instance
guard_agent = GuardAgent()

def evaluate_response(
    context: str, 
    user_question: str, 
    assistant_response: str
) -> Tuple[bool, str, str]:
    """
    Convenience function for evaluating assistant responses.
    
    Args:
        context (str): The retrieved context from RAG
        user_question (str): The user's original question
        assistant_response (str): The assistant's draft response
        
    Returns:
        Tuple[bool, str, str]: (is_approved, final_response, guard_reasoning)
    """
    return guard_agent.evaluate_response(context, user_question, assistant_response)

# Legacy functions for backward compatibility
def is_safe_response(text: str) -> bool:
    """Legacy function - returns basic safety check"""
    return guard_agent.quick_safety_check(text)

def redirect_response() -> str:
    """Legacy function - returns fallback message"""
    return DEFAULT_FALLBACK_MESSAGE
