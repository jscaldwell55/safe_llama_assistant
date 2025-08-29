# guard.py - Production-Ready Grounding Validator with Emergency Detection

import logging
import numpy as np
from typing import Optional, List, Tuple
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
# EMERGENCY MESSAGES
# ============================================================================

EMERGENCY_MESSAGE = """If this is a medical emergency, please call 911 immediately.

For poison control: 1-800-222-1222
For mental health crisis: 988 (Suicide & Crisis Lifeline)

I cannot provide emergency medical guidance. Please seek immediate professional help."""

# ============================================================================
# VALIDATION RESULTS
# ============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PERSONAL_MEDICAL = "personal_medical"
    EMERGENCY = "emergency"

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
    """Pre-screens queries for emergencies and personal medical advice"""
    
    # Emergency indicators that suggest immediate help needed
    EMERGENCY_INDICATORS = [
        'dying', 'die', 'dead', 'death',
        'emergency', '911', 'ambulance',
        'hospital', 'ER', 'emergency room',
        'unconscious', 'not breathing', 'no pulse',
        'overdose', 'overdosing', 'OD',
        'poisoned', 'poison',
        'heart attack', 'stroke',
        'seizure', 'seizing',
        'bleeding out', 'severe bleeding',
        'can\'t breathe', 'cannot breathe',
        'anaphylaxis', 'allergic reaction'
    ]
    
    @staticmethod
    def check_emergency(query: str) -> bool:
        """Check if query indicates a medical emergency"""
        query_lower = query.lower()

        logger.info(f"check_emergency called with: '{query_lower[:100]}...'")
        
        # Check for emergency indicators
        for indicator in QueryValidator.EMERGENCY_INDICATORS:
            if indicator in query_lower:
                logger.warning(f"Emergency query detected: '{query[:50]}...'")
                return True
        
        # Check for crisis patterns
        crisis_patterns = [
            'about to die',
            'going to die',
            'help me now',
            'urgent help',
            'call 911',
            'need ambulance'
        ]
        
        for pattern in crisis_patterns:
            if pattern in query_lower:
                logger.warning(f"Crisis pattern detected: '{query[:50]}...'")
                return True
                
        return False
    
    @staticmethod
    def check_personal_medical(query: str) -> bool:
        """Check if query is asking for personal medical advice"""
        query_lower = query.lower()
        
      
        
        # Allow general interaction/safety questions
        interaction_terms = [
            'interaction', 'interact with', 
            'mixing', 'combine', 'together with',
            'alcohol', 'other medication', 'other drugs',
            'food', 'grapefruit', 'caffeine'
        ]
        
        if any(term in query_lower for term in interaction_terms):
            logger.debug(f"Interaction query allowed: '{query[:50]}...'")
            return False  # Let it through to check documentation
        
        # Allow general safety/usage questions
        general_terms = [
            'what is the dose',
            'how often',
            'how to take',
            'side effects',
            'warnings',
            'contraindications',
            'is it safe for pregnant',  # General population questions
            'is it safe for elderly',
            'is it safe for children'
        ]
        
        if any(term in query_lower for term in general_terms):
            return False
        
        # Check for truly personal medical advice
        personal_medical_phrases = [
            'should i take',
            'can i take',
            'i should take',
            'my condition',
            'my symptoms', 
            'my disease',
            'i have been diagnosed',
            'my doctor',
            'in my case',
            'for me personally',
            'my medical history'
        ]
        
        if any(phrase in query_lower for phrase in personal_medical_phrases):
            logger.warning(f"Personal medical advice query detected: '{query[:50]}...'")
            return True
            
        return False
    
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate query and return (is_blocked, message)
        Returns (True, message) if query should be blocked
        Returns (False, None) if query can proceed
        """
        # Check for emergency first
        if QueryValidator.check_emergency(query):
            return True, EMERGENCY_MESSAGE
        
        # Then check for personal medical advice
        if BLOCK_PERSONAL_MEDICAL and QueryValidator.check_personal_medical(query):
            return True, PERSONAL_MEDICAL_ADVICE_MESSAGE
        
        # Query is OK to process
        return False, None

# ============================================================================
# GROUNDING VALIDATOR
# ============================================================================

class GroundingValidator:
    """
    Production-ready semantic grounding validator with safe thresholds
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        # Use safe threshold from config (0.45)
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
        Validate response grounding with production-safe threshold
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
            
            # Use production threshold
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