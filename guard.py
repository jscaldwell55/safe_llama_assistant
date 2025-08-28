# guard.py - Simplified Safety System with Grounding Only

import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from config import (
    ENABLE_GUARD,
    SEMANTIC_SIMILARITY_THRESHOLD, # Will be 0.45 now
    GUARD_FALLBACK_MESSAGE,
    NO_CONTEXT_FALLBACK_MESSAGE    # Will be used for exact comparison
)

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

class ValidationResult(Enum):
    """Validation outcomes"""
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class ValidationDecision:
    """Validation decision"""
    result: ValidationResult
    final_response: str
    grounding_score: float = 0.0
    reasoning: str = ""

# ============================================================================
# SIMPLIFIED GUARD - GROUNDING ONLY
# ============================================================================

class SimpleGroundingGuard:
    """
    Simplified guard that ONLY checks semantic similarity between response and context
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.similarity_threshold = SEMANTIC_SIMILARITY_THRESHOLD
        self.embedding_model = None
        
        logger.info(f"SimpleGroundingGuard initialized - enabled: {self.enabled}, threshold: {self.similarity_threshold}")
        
        if self.enabled:
            self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load embedding model for grounding checks"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            if self.embedding_model:
                logger.info("Embedding model loaded for grounding validation")
            else:
                logger.error("Failed to load embedding model")
                self.enabled = False
        except Exception as e:
            logger.error(f"Could not load embedding model: {e}")
            self.enabled = False
    
    def calculate_grounding_score(self, response: str, context: str) -> float:
        """Calculate semantic similarity between response and context"""
        if not self.embedding_model or not response or not context:
            return 0.0
        
        try:
            # Encode both texts
            response_embedding = self.embedding_model.encode(response, show_progress_bar=False)
            context_embedding = self.embedding_model.encode(context, show_progress_bar=False)
            
            # Normalize vectors
            response_embedding = response_embedding / (np.linalg.norm(response_embedding) + 1e-8)
            context_embedding = context_embedding / (np.linalg.norm(context_embedding) + 1e-8)
            
            # Calculate cosine similarity
            similarity = float(np.dot(response_embedding, context_embedding))
            
            logger.debug(f"Grounding score: {similarity:.3f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating grounding score: {e}")
            return 0.0
    
    def validate_response(self, response: str, context: str) -> ValidationDecision:
        """
        Validate response is grounded in context
        
        Args:
            response: Generated response from LLM
            context: Retrieved context from RAG
            
        Returns:
            ValidationDecision with result and score
        """
        
        if not self.enabled:
            logger.debug("Guard disabled - approving response")
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                grounding_score=1.0,
                reasoning="Guard disabled"
            )
        
        # Check for standard fallback messages from config (always approve these for robustness)
        # Using strip() and lower() for a slightly more flexible comparison
        if response.strip().lower() == NO_CONTEXT_FALLBACK_MESSAGE.strip().lower() or \
           response.strip().lower() == GUARD_FALLBACK_MESSAGE.strip().lower():
            logger.debug("Recognized standard fallback message - approving without further grounding check")
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                grounding_score=1.0, # Assign a high score as it's an intended fallback
                reasoning="Recognized standard fallback message"
            )
        
        # If no context, the response should ideally be the NO_CONTEXT_FALLBACK_MESSAGE
        # If it's not, we'll still reject it and provide the official fallback.
        if not context or len(context.strip()) < 50: # Check for minimal context length
            logger.warning("No sufficient context available - response should be fallback")
            # If the response is not the expected NO_CONTEXT_FALLBACK_MESSAGE, replace it
            if response.strip().lower() != NO_CONTEXT_FALLBACK_MESSAGE.strip().lower():
                final_response = NO_CONTEXT_FALLBACK_MESSAGE
                logger.debug(f"Replacing ungrounded response with '{NO_CONTEXT_FALLBACK_MESSAGE}'")
            else:
                final_response = response
            
            return ValidationDecision(
                result=ValidationResult.REJECTED, # Even if it's the fallback, it's a rejection of the query
                final_response=final_response,
                grounding_score=0.0,
                reasoning="No sufficient context available"
            )
        
        # Calculate grounding score
        grounding_score = self.calculate_grounding_score(response, context)
        
        logger.info(f"Grounding validation - Score: {grounding_score:.3f}, Threshold: {self.similarity_threshold}")
        
        # Check if response meets grounding threshold
        if grounding_score >= self.similarity_threshold:
            logger.info(f"Response APPROVED with grounding score: {grounding_score:.3f}")
            return ValidationDecision(
                result=ValidationResult.APPROVED,
                final_response=response,
                grounding_score=grounding_score,
                reasoning=f"Grounding score {grounding_score:.3f} meets threshold"
            )
        else:
            logger.warning(f"Response REJECTED - insufficient grounding: {grounding_score:.3f}")
            return ValidationDecision(
                result=ValidationResult.REJECTED,
                final_response=GUARD_FALLBACK_MESSAGE, # Always use the official guard fallback
                grounding_score=grounding_score,
                reasoning=f"Insufficient grounding score: {grounding_score:.3f}"
            )

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Single global instance
grounding_guard = SimpleGroundingGuard()

# Legacy compatibility alias
hybrid_guard = grounding_guard
persona_validator = grounding_guard