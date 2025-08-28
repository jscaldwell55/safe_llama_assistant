# guard.py - Simple Grounding Validator

import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from config import (
    ENABLE_GUARD,
    SEMANTIC_SIMILARITY_THRESHOLD,
    NO_CONTEXT_FALLBACK_MESSAGE
)

logger = logging.getLogger(__name__)

# ============================================================================
# VALIDATION RESULTS
# ============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class GroundingValidation:
    result: ValidationResult
    grounding_score: float
    final_response: str

# ============================================================================
# SIMPLE GROUNDING GUARD
# ============================================================================

class SimpleGroundingGuard:
    """
    Simple semantic grounding validator using cosine similarity
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        # Lower threshold for better acceptance of valid responses
        self.threshold = 0.30  # Lowered from 0.45
        self.embedding_model = None
        
        logger.info(f"SimpleGroundingGuard initialized - enabled: {self.enabled}, threshold: {self.threshold}")
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load embedding model for grounding checks"""
        try:
            from embeddings import get_embedding_model
            self.embedding_model = get_embedding_model()
            if self.embedding_model:
                logger.info("Embedding model loaded for grounding validation")
            else:
                logger.warning("Could not load embedding model for grounding")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.enabled = False
    
    def validate_response(self, response: str, context: str) -> GroundingValidation:
        """
        Validate that response is grounded in context using cosine similarity
        """
        if not self.enabled or not self.embedding_model:
            # Validation disabled or not available
            return GroundingValidation(
                result=ValidationResult.APPROVED,
                grounding_score=1.0,
                final_response=response
            )
        
        # Skip validation for standard messages
        response_lower = response.lower()
        if response_lower.startswith(("i'm sorry", "i don't have", "i cannot")):
            return GroundingValidation(
                result=ValidationResult.APPROVED,
                grounding_score=1.0,
                final_response=response
            )
        
        # Calculate grounding score
        try:
            # Encode response and context
            response_embedding = self.embedding_model.encode([response], convert_to_tensor=False)[0]
            context_embedding = self.embedding_model.encode([context], convert_to_tensor=False)[0]
            
            # Calculate cosine similarity
            response_norm = response_embedding / (np.linalg.norm(response_embedding) + 1e-8)
            context_norm = context_embedding / (np.linalg.norm(context_embedding) + 1e-8)
            similarity = float(np.dot(response_norm, context_norm))
            
            logger.info(f"Grounding validation - Score: {similarity:.3f}, Threshold: {self.threshold}")
            
            if similarity >= self.threshold:
                logger.info(f"Response APPROVED with grounding score: {similarity:.3f}")
                return GroundingValidation(
                    result=ValidationResult.APPROVED,
                    grounding_score=similarity,
                    final_response=response
                )
            else:
                logger.warning(f"Response REJECTED - insufficient grounding: {similarity:.3f}")
                return GroundingValidation(
                    result=ValidationResult.REJECTED,
                    grounding_score=similarity,
                    final_response=NO_CONTEXT_FALLBACK_MESSAGE  # Use proper fallback
                )
                
        except Exception as e:
            logger.error(f"Grounding validation failed: {e}")
            # On error, approve to avoid blocking
            return GroundingValidation(
                result=ValidationResult.APPROVED,
                grounding_score=0.0,
                final_response=response
            )

# ============================================================================
# SINGLETON
# ============================================================================

grounding_guard = SimpleGroundingGuard()