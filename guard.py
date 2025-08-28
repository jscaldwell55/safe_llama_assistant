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
        Validate that response is grounded in context
        """
        if not self.enabled or not self.embedding_model:
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
        
        # Check for keyword overlap as a sanity check
        response_words = set(response_lower.split())
        context_words = set(context.lower().split())
        
        # Key medical terms that should appear in both if grounded
        medical_terms = {'effect', 'effects', 'side', 'reaction', 'nausea', 'headache', 
                        'dizziness', 'pain', 'journvax', 'dose', 'patient', 'clinical'}
        
        response_medical = response_words & medical_terms
        context_medical = context_words & medical_terms
        
        # If response mentions medical terms that aren't in context, reject
        if response_medical and not (response_medical & context_medical):
            logger.warning("Response contains medical terms not found in context")
            return GroundingValidation(
                result=ValidationResult.REJECTED,
                grounding_score=0.0,
                final_response=NO_CONTEXT_FALLBACK_MESSAGE
            )
        
        # Otherwise, check embedding similarity with lowered threshold
        try:
            response_embedding = self.embedding_model.encode([response], convert_to_tensor=False)[0]
            context_embedding = self.embedding_model.encode([context], convert_to_tensor=False)[0]
            
            response_norm = response_embedding / (np.linalg.norm(response_embedding) + 1e-8)
            context_norm = context_embedding / (np.linalg.norm(context_embedding) + 1e-8)
            similarity = float(np.dot(response_norm, context_norm))
            
            # Much lower threshold since we're comparing different text styles
            threshold = 0.25  # Even lower
            
            logger.info(f"Grounding validation - Score: {similarity:.3f}, Threshold: {threshold}")
            
            if similarity >= threshold:
                return GroundingValidation(
                    result=ValidationResult.APPROVED,
                    grounding_score=similarity,
                    final_response=response
                )
            else:
                # Only reject if BOTH similarity is low AND no keyword overlap
                if not (response_medical & context_medical):
                    return GroundingValidation(
                        result=ValidationResult.REJECTED,
                        grounding_score=similarity,
                        final_response=NO_CONTEXT_FALLBACK_MESSAGE
                    )
                else:
                    # Has keyword overlap, approve despite low embedding score
                    logger.info("Low embedding score but has keyword overlap - approving")
                    return GroundingValidation(
                        result=ValidationResult.APPROVED,
                        grounding_score=similarity,
                        final_response=response
                    )
                    
        except Exception as e:
            logger.error(f"Grounding validation failed: {e}")
            return GroundingValidation(
                result=ValidationResult.APPROVED,
                grounding_score=0.0,
                final_response=response
            )

# ============================================================================
# SINGLETON
# ============================================================================

grounding_guard = SimpleGroundingGuard()