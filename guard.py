import json
import logging
from typing import Dict, Any, Tuple, List
import re
from llm_client import call_guard_agent
from prompt import format_guard_prompt
from config import DEFAULT_FALLBACK_MESSAGE, ENABLE_GUARD, EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuardAgent:
    """
    LLM-based guard agent for evaluating assistant responses.
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_message = "I can't help with that. Can we talk about something else?"
        # Initialize embedding model for semantic similarity
        self.embedding_model = None
        self._init_embedding_model()
        
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
        
        # Check contextual safety validation for medical claims
        context_validation_result = self._validate_medical_context(assistant_response, context)
        if not context_validation_result[0]:
            return False, self.fallback_message, context_validation_result[1]
        
        # Enhanced RAG grounding validation with semantic similarity
        if context and len(context.strip()) > 0:
            grounding_result = self._validate_rag_grounding(assistant_response, context)
            if not grounding_result[0]:
                logger.warning(f"RAG grounding failed: {grounding_result[1]} (score: {grounding_result[2]:.2f})")
                return False, "I don't have sufficient information in our knowledge base to answer that. Could you rephrase your question or ask about something else?", grounding_result[1]
        
        # Then check basic RAG-only validation
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
        
        enhanced_prompt = f"""Review this pharmaceutical AI assistant response for safety and compliance.

CRITICAL REQUIREMENTS:
1. NO medical advice, diagnosis, or treatment recommendations
2. NO dosage instructions beyond what's in provided context
3. NO off-label use discussions
4. NO competitor product mentions
5. ONLY FDA-approved information from context

IMMEDIATE REJECTION CRITERIA:
- Any suggestion to start, stop, or modify medication
- Specific dosing recommendations not in context
- Claims about drug efficacy not in context
- Mentions of drug interactions not in context
- Any content that could influence medical decisions
- Factual claims not supported by the provided context
- Information that appears to come from general knowledge rather than context

APPROVAL CRITERIA:
- Cites only from provided context
- Includes appropriate medical disclaimers
- Maintains professional, neutral tone
- Refers to healthcare professionals appropriately
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
        Detect pharmaceutical compliance violations with context-aware analysis.
        
        Args:
            assistant_response (str): The assistant's response to check
            
        Returns:
            Tuple[bool, str]: (violation_found, violation_details)
        """
        import re
        
        response_lower = assistant_response.lower()
        violations_found = []
        
        # Context-aware analysis: Check if response is quoting from RAG content
        rag_quote_indicators = [
            "according to our documentation",
            "based on the available information",
            "the documentation states",
            "our knowledge base shows",
            "from our available content"
        ]
        
        is_quoting_rag = any(indicator in response_lower for indicator in rag_quote_indicators)
        
        for category, detection_config in self.violation_categories.items():
            patterns = detection_config["patterns"]
            
            # Check regex patterns with context awareness
            for pattern in patterns:
                match = re.search(pattern, response_lower, re.IGNORECASE)
                if match:
                    # Analyze context around the match
                    match_context = response_lower[max(0, match.start()-50):min(len(response_lower), match.end()+50)]
                    
                    # Check if it's explaining terminology (allowed)
                    if self._is_terminology_explanation(match_context, category):
                        continue
                        
                    # Check if it's quoting from RAG (allowed)
                    if is_quoting_rag and self._is_rag_quote(match_context):
                        continue
                    
                    # Check if it's making recommendations (rejected)
                    if self._is_recommendation(match_context, category):
                        violations_found.append({
                            "category": category,
                            "pattern": pattern,
                            "type": "recommendation_detected"
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
            # Check for directive language about medications with context
            directive_phrases = [
                "should take", "need to take", "must take", "start taking",
                "stop taking", "increase dose", "decrease dose"
            ]
            
            for phrase in directive_phrases:
                if phrase in response_lower:
                    # Get context around the phrase
                    phrase_index = response_lower.find(phrase)
                    context = response_lower[max(0, phrase_index-50):min(len(response_lower), phrase_index+100)]
                    
                    # Check if it's quoting documentation (allowed)
                    if any(indicator in context for indicator in ["documentation states", "according to", "guidelines say"]):
                        continue
                        
                    # Check if it's conditional/hypothetical (allowed)
                    if any(indicator in context for indicator in ["if your doctor", "when prescribed", "as directed by"]):
                        continue
                    
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
    
    def _is_terminology_explanation(self, context: str, category: str) -> bool:
        """Check if the context is explaining terminology (allowed)"""
        explanation_indicators = [
            "means", "refers to", "is defined as", "is a term for",
            "is called", "known as", "terminology", "in medical terms"
        ]
        return any(indicator in context for indicator in explanation_indicators)
    
    def _is_rag_quote(self, context: str) -> bool:
        """Check if the context appears to be quoting from RAG content"""
        quote_indicators = [
            "states", "says", "indicates", "shows", "mentions",
            "according to", "based on", "per the", "as stated"
        ]
        return any(indicator in context for indicator in quote_indicators)
    
    def _is_recommendation(self, context: str, category: str) -> bool:
        """Check if the context is making a recommendation (should be rejected)"""
        if category == "unlicensed_medical_advice":
            recommendation_phrases = [
                "you should", "i recommend", "i suggest", "you need to",
                "you must", "it's best to", "try taking", "consider taking"
            ]
            return any(phrase in context for phrase in recommendation_phrases)
        return False
    
    def _validate_medical_context(self, response: str, context: str) -> Tuple[bool, str]:
        """Validate medical claims against provided context"""
        
        # Extract medical claims from response
        medical_claims = self._extract_medical_claims(response)
        
        # If no medical claims, validation passes
        if not medical_claims:
            return True, "No medical claims to validate"
        
        # If no context provided but medical claims exist, reject
        if not context or len(context.strip()) == 0:
            return False, "Medical claims made without supporting context"
        
        context_lower = context.lower()
        
        for claim in medical_claims:
            # Check if claim is supported by context
            if not self._claim_supported_by_context(claim, context_lower):
                return False, f"Unsupported medical claim: {claim['claim']}"
        
        return True, "Medical claims validated against context"
    
    def _extract_medical_claims(self, response: str) -> List[Dict[str, str]]:
        """Extract medical claims from the response"""
        import re
        
        response_lower = response.lower()
        medical_claims = []
        
        # Patterns that indicate medical claims
        claim_patterns = [
            # Efficacy claims
            (r'(effective|works|helps|treats|reduces|improves|prevents)\s+(?:for|with|against)?\s*([^.]+)', 'efficacy'),
            # Side effect claims
            (r'(side effects?|adverse effects?|reactions?)\s+(?:include|are|may be)?\s*([^.]+)', 'side_effect'),
            # Dosage claims
            (r'(\d+\s*(?:mg|ml|mcg|iu|units?))\s+(?:daily|per day|twice|three times)', 'dosage'),
            # Interaction claims
            (r'(interacts?|interaction)\s+with\s+([^.]+)', 'interaction'),
            # Indication claims
            (r'(?:used|prescribed|indicated)\s+(?:for|to treat)\s+([^.]+)', 'indication'),
            # Duration claims
            (r'(takes?|requires?)\s+(\d+\s*(?:days?|weeks?|months?))', 'duration'),
            # Contraindication claims
            (r'(?:should not|must not|avoid|contraindicated)\s+(?:if|when|in)\s+([^.]+)', 'contraindication')
        ]
        
        for pattern, claim_type in claim_patterns:
            matches = re.finditer(pattern, response_lower, re.IGNORECASE)
            for match in matches:
                claim_text = match.group(0)
                # Get context around the claim
                start = max(0, match.start() - 20)
                end = min(len(response), match.end() + 20)
                claim_context = response[start:end]
                
                medical_claims.append({
                    'claim': claim_text,
                    'type': claim_type,
                    'context': claim_context
                })
        
        return medical_claims
    
    def _claim_supported_by_context(self, claim: Dict[str, str], context_lower: str) -> bool:
        """Check if a medical claim is supported by the provided context"""
        
        claim_text = claim['claim'].lower()
        claim_type = claim['type']
        
        # Extract key terms from the claim
        key_terms = self._extract_key_terms(claim_text, claim_type)
        
        # Check if key terms appear in context
        terms_found = 0
        for term in key_terms:
            if term in context_lower:
                terms_found += 1
        
        # Require at least 60% of key terms to be in context
        if len(key_terms) > 0:
            match_ratio = terms_found / len(key_terms)
            return match_ratio >= 0.6
        
        # If no key terms extracted, check if full claim appears in context
        return claim_text in context_lower
    
    def _extract_key_terms(self, claim_text: str, claim_type: str) -> List[str]:
        """Extract key terms from a medical claim"""
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                     'that', 'these', 'those', 'if', 'when', 'where', 'which', 'who'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', claim_text.lower())
        
        # Filter out stop words and short words
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add specific medical terms based on claim type
        if claim_type == 'dosage':
            # Extract dosage numbers and units
            dosage_matches = re.findall(r'\d+\s*(?:mg|ml|mcg|iu|units?)', claim_text)
            key_terms.extend(dosage_matches)
        elif claim_type == 'efficacy':
            # Focus on condition/symptom terms
            key_terms = [t for t in key_terms if t not in ['effective', 'works', 'helps', 'treats']]
        elif claim_type == 'side_effect':
            # Focus on the actual side effects, not the phrase "side effects"
            key_terms = [t for t in key_terms if t not in ['side', 'effects', 'effect', 'adverse']]
        
        return list(set(key_terms))  # Remove duplicates
    
    def _init_embedding_model(self):
        """Initialize the embedding model for semantic similarity"""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Initialized embedding model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def _validate_rag_grounding(self, response: str, context: str) -> Tuple[bool, str, float]:
        """
        Validate response grounding with semantic similarity confidence score
        Returns: (is_valid, reasoning, confidence_score)
        """
        
        # If no embedding model, fall back to existing validation
        if not self.embedding_model:
            return True, "Embedding model not available, using basic validation", 0.0
        
        # Extract factual statements from response
        factual_statements = self._extract_factual_claims(response)
        
        # If no factual statements, response is likely conversational
        if not factual_statements:
            return True, "No factual claims requiring grounding", 1.0
        
        # If no context but factual statements exist
        if not context or len(context.strip()) == 0:
            return False, "Factual claims made without context", 0.0
        
        grounding_scores = []
        ungrounded_statements = []
        
        for statement in factual_statements:
            # Calculate semantic similarity with context
            score = self._calculate_grounding_score(statement, context)
            grounding_scores.append(score)
            
            if score < 0.7:  # Threshold for individual statement grounding
                ungrounded_statements.append(statement)
        
        avg_score = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0
        
        if avg_score < 0.7:  # Overall threshold for grounding
            reasoning = f"Insufficient grounding in context (avg score: {avg_score:.2f}). Ungrounded: {'; '.join(ungrounded_statements[:2])}"
            return False, reasoning, avg_score
        
        return True, f"Well-grounded response (avg score: {avg_score:.2f})", avg_score
    
    def _extract_factual_claims(self, response: str) -> List[str]:
        """Extract factual claims from response (not just medical claims)"""
        import re
        
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        
        factual_statements = []
        
        # Patterns indicating factual claims
        factual_indicators = [
            r'\b(?:is|are|was|were)\s+\w+',  # "X is Y" statements
            r'\b(?:contains?|includes?|has|have)\s+\w+',  # Compositional claims
            r'\b(?:causes?|results?\s+in|leads?\s+to)\s+\w+',  # Causal claims
            r'\b(?:works?\s+by|functions?\s+through)\s+\w+',  # Mechanism claims
            r'\b\d+\s*(?:percent|%|mg|ml|mcg|iu|units?)\b',  # Quantitative claims
            r'\b(?:approved|indicated|prescribed)\s+(?:for|to)\b',  # Medical use claims
            r'\b(?:studies?|research|trials?|data)\s+(?:show|indicate|suggest)\b',  # Research claims
            r'\b(?:typically|usually|commonly|often|always|never)\s+\w+',  # Frequency claims
            r'\b(?:effective|ineffective|safe|unsafe)\s+(?:for|in|against)\b',  # Efficacy claims
        ]
        
        # Exclude conversational patterns
        conversational_patterns = [
            r'^(?:hello|hi|hey|thanks|thank you|you\'re welcome)',
            r'^\s*(?:i\'m sorry|i don\'t|i can\'t|i\'d be happy)',
            r'^\s*(?:would you like|do you have|is there|can i help)',
            r'^\s*(?:let me|i\'ll|here\'s what)',
            r'^\s*(?:based on|according to|from our|in our)'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Check if it's conversational
            is_conversational = any(re.search(pattern, sentence.lower()) for pattern in conversational_patterns)
            if is_conversational:
                continue
            
            # Check if it contains factual indicators
            contains_factual = any(re.search(pattern, sentence.lower()) for pattern in factual_indicators)
            
            # Also include sentences with specific medical/technical terms
            contains_technical = bool(re.search(r'\b(?:medication|drug|treatment|therapy|symptoms?|conditions?|diseases?|disorders?)\b', sentence.lower()))
            
            if contains_factual or contains_technical:
                factual_statements.append(sentence)
        
        return factual_statements
    
    def _calculate_grounding_score(self, statement: str, context: str) -> float:
        """Calculate semantic similarity between statement and context"""
        try:
            # Encode statement and context
            statement_embedding = self.embedding_model.encode(statement, convert_to_tensor=False)
            
            # Split context into sentences for more granular matching
            context_sentences = re.split(r'[.!?]+', context)
            context_sentences = [s.strip() for s in context_sentences if len(s.strip()) > 20]
            
            if not context_sentences:
                # If no good sentences, use whole context
                context_embedding = self.embedding_model.encode(context, convert_to_tensor=False)
                return float(np.dot(statement_embedding, context_embedding) / 
                           (np.linalg.norm(statement_embedding) * np.linalg.norm(context_embedding)))
            
            # Calculate similarity with each context sentence
            max_similarity = 0.0
            for ctx_sentence in context_sentences:
                ctx_embedding = self.embedding_model.encode(ctx_sentence, convert_to_tensor=False)
                
                # Cosine similarity
                similarity = float(np.dot(statement_embedding, ctx_embedding) / 
                                 (np.linalg.norm(statement_embedding) * np.linalg.norm(ctx_embedding)))
                
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            return 0.0

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
