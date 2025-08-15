# guard.py - Simplified Validation for Persona-Based Architecture

import logging
import re
import json
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

from config import (
    ENABLE_GUARD,
    SEMANTIC_SIMILARITY_THRESHOLD,
    USE_LLM_GUARD,
    DEFAULT_FALLBACK_MESSAGE
)

logger = logging.getLogger(__name__)

# ============================================================================
# VALIDATION TYPES AND STRUCTURES
# ============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

@dataclass
class ValidationDecision:
    result: ValidationResult
    final_response: str
    reasoning: str
    confidence: float = 0.0
    modifications: List[str] = None
    metadata: Dict[str, Any] = None

# ============================================================================
# STREAMLINED VALIDATOR
# ============================================================================

class PersonaValidator:
    """
    Simplified validator for the Persona-based architecture.
    Since personas already enforce constraints, validation is lighter.
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.use_llm = USE_LLM_GUARD
        self.sim_threshold = SEMANTIC_SIMILARITY_THRESHOLD
        self.embedding_model = self._load_embedding_model()
        
        # Critical patterns that still need checking
        self.critical_patterns = {
            "dosage_directive": r'\b(?:take|consume|inject|use)\s+\d+\s*(?:mg|ml|mcg)\b',
            "medical_imperative": r'\b(?:you must|you should|never take|always take|stop taking)\b',
            "diagnostic": r'\b(?:you have|you are suffering from|diagnosed with)\b',
            "dangerous_route": r'\b(?:crush|snort|inject|smoke|dissolve)\b'
        }
    
    def _load_embedding_model(self):
        """Load embedding model for semantic similarity"""
        try:
            from embeddings import get_embedding_model
            return get_embedding_model()
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            return None
    
    async def validate_response(
        self,
        response: str,
        strategy_used: str,
        components: Optional[Dict[str, Any]] = None,
        context: str = "",
        query: str = ""
    ) -> ValidationDecision:
        """
        Validate a response based on the strategy used to generate it
        """
        if not self.enabled:
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation disabled",
                confidence=1.0
            )
        
        try:
            # Strategy-specific validation
            if strategy_used == "pure_empathy":
                # Pure empathy responses don't need grounding checks
                return await self._validate_empathy_only(response)
                
            elif strategy_used == "pure_facts":
                # Pure facts need strict grounding
                return await self._validate_facts_only(response, context)
                
            elif strategy_used == "synthesized":
                # Synthesized responses need component-aware validation
                return await self._validate_synthesized(
                    response, components, context, query
                )
                
            else:  # conversational
                # Light validation for conversational responses
                return await self._validate_conversational(response)
                
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=DEFAULT_FALLBACK_MESSAGE,
                reasoning=f"Validation error: {str(e)}",
                confidence=0.0
            )
    
    async def _validate_empathy_only(self, response: str) -> ValidationDecision:
        """
        Validate pure empathetic responses.
        These should NOT contain medical facts.
        """
        response_lower = response.lower()
        
        # Check for leaked medical information
        medical_terms = [
            r'\b\d+\s*(?:mg|ml|mcg)\b',  # Dosages
            r'\b(?:lexapro|escitalopram|ssri)\b',  # Drug names
            r'\b(?:side effect|adverse|interaction)\b',  # Medical terms
            r'\b(?:indication|contraindication)\b'
        ]
        
        for pattern in medical_terms:
            if re.search(pattern, response_lower):
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response="I understand your concerns. I'm here to help you find the information you need.",
                    reasoning="Empathetic response contained medical information",
                    confidence=0.9
                )
        
        # Check for dangerous patterns
        for pattern_name, pattern in self.critical_patterns.items():
            if re.search(pattern, response_lower):
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response="I understand you're looking for help. Please consult with a healthcare professional.",
                    reasoning=f"Critical pattern detected: {pattern_name}",
                    confidence=0.95
                )
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="Pure empathy response validated",
            confidence=0.9
        )
    
    async def _validate_facts_only(self, response: str, context: str) -> ValidationDecision:
        """
        Validate pure factual responses.
        These need strict grounding in context.
        """
        if not context or not context.strip():
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response="I don't have that information in the documentation.",
                reasoning="No context available for factual response",
                confidence=0.95
            )
        
        # Calculate grounding score
        grounding_score = self._calculate_grounding_score(response, context)
        
        if grounding_score < self.sim_threshold:
            # Try lexical grounding as fallback
            if not self._lexical_grounding_check(response, context):
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response="I don't have sufficient information about that in the documentation.",
                    reasoning=f"Poor grounding score: {grounding_score:.2f}",
                    confidence=0.8
                )
        
        # Check for critical patterns even in grounded responses
        response_lower = response.lower()
        for pattern_name, pattern in self.critical_patterns.items():
            if re.search(pattern, response_lower):
                # For facts, check if the pattern is actually in the context
                if not re.search(pattern, context.lower()):
                    return ValidationDecision(
                        result=ValidationResult.REJECTED,
                        final_response="I cannot provide that type of information.",
                        reasoning=f"Critical pattern not grounded: {pattern_name}",
                        confidence=0.9
                    )
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning=f"Factual response grounded (score: {grounding_score:.2f})",
            confidence=min(0.95, grounding_score + 0.3)
        )
    
    async def _validate_synthesized(
        self,
        response: str,
        components: Optional[Dict[str, Any]],
        context: str,
        query: str
    ) -> ValidationDecision:
        """
        Validate synthesized responses.
        Only the factual component needs grounding.
        """
        # If we have components, validate them separately
        if components:
            facts_component = components.get("facts_component", "")
            
            if facts_component and facts_component != "No relevant information found in documentation.":
                # Validate just the facts component
                grounding_score = self._calculate_grounding_score(facts_component, context)
                
                if grounding_score < self.sim_threshold:
                    return ValidationDecision(
                        result=ValidationResult.REJECTED,
                        final_response=self._create_safe_fallback(query),
                        reasoning=f"Facts component poorly grounded: {grounding_score:.2f}",
                        confidence=0.8
                    )
        
        # Check the full response for critical patterns
        response_lower = response.lower()
        for pattern_name, pattern in self.critical_patterns.items():
            if re.search(pattern, response_lower):
                # For synthesized responses, be more lenient if it's quoting context
                if context and re.search(pattern, context.lower()):
                    continue  # Pattern is in context, allow it
                    
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=self._create_safe_fallback(query),
                    reasoning=f"Critical pattern in synthesis: {pattern_name}",
                    confidence=0.85
                )
        
        # LLM validation if enabled and needed
        if self.use_llm and self._needs_llm_validation(response):
            llm_decision = await self._llm_safety_check(response, context, query)
            if llm_decision.result == ValidationResult.REJECTED:
                return llm_decision
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="Synthesized response validated",
            confidence=0.85,
            metadata={"synthesis_validation": "component-aware"}
        )
    
    async def _validate_conversational(self, response: str) -> ValidationDecision:
        """
        Light validation for conversational responses
        """
        response_lower = response.lower()
        
        # Only check for the most critical patterns
        critical_checks = ["diagnostic", "dangerous_route"]
        for pattern_name in critical_checks:
            if pattern_name in self.critical_patterns:
                pattern = self.critical_patterns[pattern_name]
                if re.search(pattern, response_lower):
                    return ValidationDecision(
                        result=ValidationResult.REJECTED,
                        final_response="I can help you with that. What would you like to know?",
                        reasoning=f"Critical pattern in conversation: {pattern_name}",
                        confidence=0.9
                    )
        
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=response,
            reasoning="Conversational response approved",
            confidence=0.95
        )
    
    def _calculate_grounding_score(self, text: str, context: str) -> float:
        """Calculate semantic similarity score between text and context"""
        if not context or not self.embedding_model:
            return 0.0
        
        try:
            text_embedding = self.embedding_model.encode(
                text, convert_to_tensor=False, show_progress_bar=False
            )
            context_embedding = self.embedding_model.encode(
                context, convert_to_tensor=False, show_progress_bar=False
            )
            
            similarity = float(
                np.dot(text_embedding, context_embedding) / 
                (np.linalg.norm(text_embedding) * np.linalg.norm(context_embedding))
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            return 0.0
    
    def _lexical_grounding_check(self, text: str, context: str) -> bool:
        """Simple lexical overlap check as fallback"""
        if not context:
            return False
            
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        text_words -= stop_words
        context_words -= stop_words
        
        if not text_words:
            return True
            
        overlap = len(text_words & context_words) / len(text_words)
        return overlap > 0.5
    
    def _needs_llm_validation(self, response: str) -> bool:
        """Determine if LLM validation is needed"""
        # Skip LLM for very short responses
        if len(response) < 50:
            return False
        
        # Skip for responses that are clearly just lists
        if response.count('\n') > 5 or response.count('•') > 3:
            return False
        
        # Need LLM for complex responses
        return len(response) > 200
    
    async def _llm_safety_check(
        self,
        response: str,
        context: str,
        query: str
    ) -> ValidationDecision:
        """Use LLM for safety validation"""
        try:
            from prompts import format_validation_prompt
            from llm_client import call_guard_agent
            
            prompt = format_validation_prompt(context, query, response)
            llm_response = await call_guard_agent(prompt)
            
            # Parse LLM response
            return self._parse_llm_validation(llm_response, response)
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            # Don't reject on LLM failure, just log
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="LLM validation failed, defaulting to approve",
                confidence=0.5
            )
    
    def _parse_llm_validation(self, llm_response: str, original_response: str) -> ValidationDecision:
        """Parse LLM validation response"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                verdict = data.get("verdict", "APPROVE")
                confidence = float(data.get("confidence", 0.7))
                
                if verdict == "REJECT":
                    return ValidationDecision(
                        result=ValidationResult.REJECTED,
                        final_response=DEFAULT_FALLBACK_MESSAGE,
                        reasoning=f"LLM rejection: {data.get('issues', [])}",
                        confidence=confidence
                    )
                elif verdict == "NEEDS_MODIFICATION":
                    # Could implement modification logic here
                    return ValidationDecision(
                        result=ValidationResult.MODIFIED,
                        final_response=original_response,  # For now, keep original
                        reasoning="LLM suggested modifications",
                        confidence=confidence,
                        modifications=data.get("issues", [])
                    )
        except Exception as e:
            logger.warning(f"Failed to parse LLM validation JSON: {e}")
        
        # Default to approval if parsing fails
        return ValidationDecision(
            result=ValidationResult.APPROVED,
            final_response=original_response,
            reasoning="LLM validation parsed as approval",
            confidence=0.7
        )
    
    def _create_safe_fallback(self, query: str) -> str:
        """Create a safe fallback response based on query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["worried", "scared", "anxious"]):
            return "I understand your concerns. For medical guidance, please consult with your healthcare provider."
        elif "?" in query:
            return "I don't have sufficient information about that in the documentation."
        else:
            return DEFAULT_FALLBACK_MESSAGE

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

class HybridGuardAgent:
    """Legacy compatibility wrapper"""
    
    def __init__(self):
        self.validator = PersonaValidator()
        self.enabled = ENABLE_GUARD
        self.fallback_messages = {
            "default": DEFAULT_FALLBACK_MESSAGE,
            "no_context": "I don't have that information in the documentation.",
            "unsafe_medical": "Please consult with a healthcare professional.",
        }
    
    def evaluate_response(
        self,
        context: str,
        user_question: str,
        assistant_response: str,
        conversation_history: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """Legacy synchronous interface"""
        import asyncio
        
        # Create async validation call
        async def validate():
            return await self.validator.validate_response(
                response=assistant_response,
                strategy_used="unknown",  # Legacy doesn't know strategy
                context=context,
                query=user_question
            )
        
        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        decision = loop.run_until_complete(validate())
        
        # Convert to legacy format
        is_approved = decision.result == ValidationResult.APPROVED
        return is_approved, decision.final_response, decision.reasoning

# Global instances
guard_agent = HybridGuardAgent()
persona_validator = PersonaValidator()

def evaluate_response(
    context: str,
    user_question: str,
    assistant_response: str,
    conversation_history: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Legacy global function"""
    return guard_agent.evaluate_response(
        context, user_question, assistant_response, conversation_history
    )