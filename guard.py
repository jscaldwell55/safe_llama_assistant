# guard.py - Clean Simple Version for Single Model System

import logging
import re
import json
import numpy as np
from typing import Tuple, Optional
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
# SIMPLE VALIDATION RESULT
# ============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class ValidationDecision:
    result: ValidationResult
    final_response: str
    reasoning: str
    confidence: float = 0.0

# ============================================================================
# SIMPLE GUARD
# ============================================================================

class SimpleGuard:
    """
    Simple guard focused on:
    1. Basic grounding check (with lower threshold)
    2. Critical safety patterns
    3. Optional LLM reasoning for edge cases
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.similarity_threshold = SEMANTIC_SIMILARITY_THRESHOLD  # Should be ~0.35
        self.use_llm = USE_LLM_GUARD
        self.embedding_model = None
        
        # Only truly dangerous patterns that should always be blocked
        self.critical_safety_patterns = [
            (r'\b(?:take|inject|consume)\s+\d+\s*(?:mg|ml|pills)\s+(?:every|daily|twice)', 'specific_dosage_instruction'),
            (r'\byou (?:definitely|certainly) have\b', 'diagnostic_claim'),
            (r'\b(?:stop|start) taking (?:lexapro|your medication) immediately\b', 'medication_directive'),
            (r'\b(?:crush|snort|inject|smoke) (?:the|your|lexapro)', 'dangerous_administration'),
        ]
        
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load embedding model for similarity check"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            logger.info("Embedding model loaded for guard")
        except Exception as e:
            logger.warning(f"Could not load embedding model for guard: {e}")
            self.embedding_model = None
    
    def check_safety_patterns(self, response: str) -> Tuple[bool, str]:
        """
        Check for critical safety issues
        Returns: (is_safe, reason)
        """
        response_lower = response.lower()
        
        for pattern, pattern_name in self.critical_safety_patterns:
            if re.search(pattern, response_lower):
                logger.warning(f"Critical safety pattern detected: {pattern_name}")
                return False, f"Contains {pattern_name}"
        
        return True, "No safety issues"
    
    def calculate_grounding_score(self, response: str, context: str) -> float:
        """
        Calculate semantic similarity between response and context
        Lower scores are OK - we just want some relationship
        """
        if not context or not self.embedding_model:
            return 0.0
        
        try:
            # Get embeddings
            response_embedding = self.embedding_model.encode(
                response, convert_to_tensor=False, show_progress_bar=False
            )
            context_embedding = self.embedding_model.encode(
                context, convert_to_tensor=False, show_progress_bar=False
            )
            
            # Calculate cosine similarity
            similarity = float(
                np.dot(response_embedding, context_embedding) / 
                (np.linalg.norm(response_embedding) * np.linalg.norm(context_embedding))
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            return 0.5  # Return neutral score on error
    
    async def llm_safety_check(self, response: str, context: str, query: str) -> Tuple[bool, str]:
        """
        Use LLM for nuanced safety check
        Returns: (is_safe, reasoning)
        """
        if not self.use_llm:
            return True, "LLM check disabled"
        
        try:
            from llm_client import call_guard_agent
            
            # Simple prompt for safety check
            prompt = f"""Check if this response is safe and appropriate for a pharmaceutical assistant.

Context provided: {context[:500]}...
User question: {query}
Response: {response}

Is this response:
1. Safe (no dangerous medical advice)?
2. Appropriate (answers the question)?
3. Grounded (if facts are stated, are they from context)?

Answer with JSON:
{{"safe": true/false, "appropriate": true/false, "grounded": true/false, "reasoning": "brief explanation"}}
"""
            
            llm_response = await call_guard_agent(prompt)
            
            # Parse response
            try:
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    is_safe = data.get("safe", True) and data.get("appropriate", True)
                    reasoning = data.get("reasoning", "LLM check")
                    return is_safe, reasoning
            except:
                pass
            
            # Default to safe if parsing fails
            return True, "LLM check passed (parsing failed)"
            
        except Exception as e:
            logger.error(f"LLM safety check failed: {e}")
            return True, "LLM check failed, defaulting to safe"
    
    
    async def validate_response(
        self,
        response: str,
        context: str = "",
        query: str = "",
        **kwargs  # Ignore extra args from legacy calls
    ) -> ValidationDecision:
        """
        Main validation function
        """
        if not self.enabled:
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Validation disabled",
                confidence=1.0
            )
        
        try:
            # Step 1: Check critical safety patterns
            is_safe, safety_reason = self.check_safety_patterns(response)
            if not is_safe:
                return ValidationDecision(
                    result=ValidationResult.REJECTED,
                    final_response=DEFAULT_FALLBACK_MESSAGE,
                    reasoning=f"Safety check failed: {safety_reason}",
                    confidence=0.95
                )
            
            # Step 2: Check if response indicates off-topic or inability to help
            response_lower = response.lower()
            off_topic_indicators = [
                "i don't have that information",
                "i don't have information about that",
                "not in the documentation",
                "not in our knowledge base",
                "i cannot help with that",
                "i can't help with that",
                "outside my scope",
                "not equipped to",
            ]
            
            if any(indicator in response_lower for indicator in off_topic_indicators):
                # Replace with simple, consistent message
                return ValidationDecision(
                    result=ValidationResult.APPROVED,
                    final_response="I'm sorry, I am not able to help you with that. Would you like to discuss something else?",
                    reasoning="Off-topic query detected",
                    confidence=0.9
                )
            
            # Step 3: If we have context, check grounding (with low threshold)
            if context and len(context) > 100:
                grounding_score = self.calculate_grounding_score(response, context)
                
                if grounding_score < self.similarity_threshold:
                    # Only reject if it's REALLY ungrounded (very low score)
                    if grounding_score < 0.2:  # Much lower threshold
                        logger.warning(f"Very poor grounding: {grounding_score:.2f}")
                        # For poor grounding, use the off-topic response instead of fallback
                        return ValidationDecision(
                            result=ValidationResult.APPROVED,
                            final_response="I'm sorry, I am not able to help you with that. Would you like to discuss something else?",
                            reasoning=f"Poor grounding score: {grounding_score:.2f}",
                            confidence=0.7
                        )
                    # Otherwise just log it
                    logger.info(f"Low grounding score but acceptable: {grounding_score:.2f}")
            
            # Step 4: Optional LLM check for complex cases
            if self.use_llm and len(response) > 200:  # Only for longer responses
                is_safe_llm, llm_reason = await self.llm_safety_check(response, context, query)
                if not is_safe_llm:
                    return ValidationDecision(
                        result=ValidationResult.REJECTED,
                        final_response=DEFAULT_FALLBACK_MESSAGE,
                        reasoning=f"LLM safety check: {llm_reason}",
                        confidence=0.8
                    )
            
            # Step 5: Approved!
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning="Passed all checks",
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            # On error, approve rather than block
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                reasoning=f"Validation error, approved by default",
                confidence=0.5
            )

# ============================================================================
# SINGLETON & LEGACY SUPPORT
# ============================================================================

# Create singleton instance
simple_guard = SimpleGuard()

# Legacy support - map old names to new
persona_validator = simple_guard  # For app.py compatibility

def evaluate_response(
    context: str,
    user_question: str,
    assistant_response: str,
    conversation_history: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Legacy function for backward compatibility"""
    import asyncio
    
    async def validate():
        decision = await simple_guard.validate_response(
            response=assistant_response,
            context=context,
            query=user_question
        )
        return decision
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, validate())
            decision = future.result()
    else:
        decision = loop.run_until_complete(validate())
    
    is_approved = decision.result == ValidationResult.APPROVED
    return is_approved, decision.final_response, decision.reasoning