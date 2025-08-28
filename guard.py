# guard.py - Production-Ready Grounding Validator with Safe Thresholds

import logging
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from config import (
    ENABLE_GUARD,
    SEMANTIC_SIMILARITY_THRESHOLD,
    NO_CONTEXT_FALLBACK_MESSAGE,
    PERSONAL_MEDICAL_ADVICE_MESSAGE,
    BLOCK_PERSONAL_MEDICAL,
    PERSONAL_INDICATORS,
    MEDICAL_CONTEXTS
)

logger = logging.getLogger(__name__)

# ============================================================================
# VALIDATION RESULTS
# ============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PERSONAL_MEDICAL = "personal_medical"

@dataclass
class GroundingValidation:
    result: ValidationResult
    grounding_score: float
    final_response: str
    reasoning: str = ""

# ============================================================================
# QUERY VALIDATOR
# ============================================================================

class QueryValidator:
    """Pre-screens queries for personal medical advice"""
    
    @staticmethod
    def check_personal_medical(query: str) -> bool:
        """Check if query is asking for personal medical advice"""
        if not BLOCK_PERSONAL_MEDICAL:
            return False
            
        query_lower = query.lower()
        
        # Check for personal indicators
        has_personal = any(indicator in query_lower for indicator in PERSONAL_INDICATORS)
        
        # Check for medical context
        has_medical = any(context in query_lower for context in MEDICAL_CONTEXTS)
        
        if has_personal and has_medical:
            logger.warning(f"Personal medical advice query detected: '{query[:50]}...'")
            return True
            
        return False

# ============================================================================
# GROUNDING VALIDATOR
# ============================================================================

class GroundingValidator:
    """
    Production-ready semantic grounding validator with safe thresholds
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        # Use safe threshold from config (0.75)
        self.threshold = SEMANTIC_SIMILARITY_THRESHOLD
        self.embedding_model = None
        
        logger.info(f"GroundingValidator initialized - enabled: {self.enabled}, threshold: {self.threshold}")
        
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
        Validate response grounding with production-safe threshold (0.75)
        """
        if not self.enabled or not self.embedding_model:
            # Validation disabled - approve with warning
            logger.warning("Grounding validation disabled - approving without check")
            return GroundingValidation(
                result=ValidationResult.APPROVED,
                grounding_score=0.0,
                final_response=response,
                reasoning="Validation disabled"
            )
        
        # Skip validation for standard messages
        response_lower = response.lower()
        if response_lower.startswith(("i'm sorry", "i don't have", "i cannot")):
            return GroundingValidation(
                result=ValidationResult.APPROVED,
                grounding_score=1.0,
                final_response=response,
                reasoning="Standard fallback message"
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
            
            # Use production threshold (0.75)
            if similarity >= self.threshold:
                logger.info(f"Response APPROVED with grounding score: {similarity:.3f}")
                return GroundingValidation(
                    result=ValidationResult.APPROVED,
                    grounding_score=similarity,
                    final_response=response,
                    reasoning=f"Strong grounding (score: {similarity:.3f})"
                )
            else:
                logger.warning(f"Response REJECTED - insufficient grounding: {similarity:.3f} < {self.threshold}")
                return GroundingValidation(
                    result=ValidationResult.REJECTED,
                    grounding_score=similarity,
                    final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                    reasoning=f"Weak grounding (score: {similarity:.3f} < {self.threshold})"
                )
                
        except Exception as e:
            logger.error(f"Grounding validation failed: {e}")
            # On error, reject for safety
            return GroundingValidation(
                result=ValidationResult.REJECTED,
                grounding_score=0.0,
                final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                reasoning=f"Validation error: {str(e)}"
            )

# ============================================================================
# SINGLETONS
# ============================================================================

query_validator = QueryValidator()
grounding_validator = GroundingValidator()

# Legacy compatibility
grounding_guard = grounding_validator