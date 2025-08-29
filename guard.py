# guard.py - Grounding Validator with Crisis Detection

import logging
import re
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from config import (
    ENABLE_GUARD,
    SEMANTIC_SIMILARITY_THRESHOLD,
    NO_CONTEXT_FALLBACK_MESSAGE,
    CRISIS_RESPONSE_MESSAGE
)

logger = logging.getLogger(__name__)

# ============================================================================
# CRISIS DETECTION PATTERNS
# ============================================================================

CRISIS_REGEX = {
    # Tier 1: Explicit, self-referential ideation or intent (any match => CRISIS)
    "HIGH": [
        r"\bkill\s*myself\b",
        r"\bend\s*my\s*life\b",
        r"\btake\s*my\s*life\b",
        r"\b(?:commit|completed?)\s+suicide\b",
        r"\bi\s*(?:am|m|feel|felt|was|'m)\s*suicidal\b",
        r"\bi\s*(?:want|plan|intend|aim|am going)\s*to\s*(?:kill myself|end my life|take my life|die)\b",
        r"\b(?:self[-\s]*harm|self[-\s]*injur(?:y|e|ing))\b(?!.{0,20}\b(prevent|help|support|recover|recovery|stop)\b)",
        r"\bi\s*(?:should|might|could)\s*(?:just\s*)?(?:die|disappear)\b",
        # Slang that strongly implies self-harm (self-referential)
        r"\b(?:unalive)\s*(?:myself)?\b",
        r"\boff\s*myself\b",
        r"\bkys\b",  # treat as HIGH in consumer chat; adjust if too noisy
    ],
    # Tier 2: Methods / planning / procurement language (medium confidence; boost score)
    "MEDIUM": [
        r"\b(?:how|best|easiest|quickest|cleanest)\s+(?:to|way to)\s+(?:die|kill myself|end my life|commit suicide)\b",
        r"\b(?:plan|planning|deciding)\s+to\s+(?:kill myself|end my life|die)\b",
        r"\b(?:overdos(?:e|ing)|OD)\b",
        r"\b(?:cut|slit)\s*(?:my|the)\s*(?:wrist|wrists)\b",
        r"\b(?:hang|noose|asphyxiat|suffocat|carbon\s*monoxide)\b",
        r"\b(?:jump|bridge|cliff)\b.*\b(?:die|end it|end my life)\b",
        r"\b(?:lethal\s*dose|ld50|poison)\b",
        r"\b(?:bleach|drain\s*cleaner)\b.*\b(?:drink|ingest)\b",
        r"\b(?:helium|inert\s*gas)\b.*\b(?:bag|hood)\b",
        r"\b(?:die|death)\b.*\b(?:painless|quick|fast|easiest)\b",
    ],
    # Tier 3: Implicit ideation (catch euphemisms; combine with context or score)
    "LOW": [
        r"\bi\s*(?:don'?t|do\s*not)\s*want\s*to\s*live\b",
        r"\bi\s*(?:can'?t|cannot)\s*(?:go\s*on|do\s*this\s*anymore|keep\s*(?:going|living))\b",
        r"\bi\s*wish\s+i\s*(?:were|was)\s*dead\b",
        r"\blife\s*(?:isn'?t|is\s*not)\s*worth\s*living\b",
        r"\bend\s*it\s*all\b",
        r"\bno\s*reason\s*to\s*live\b",
        r"\bbetter\s*off\s*dead\b",
        r"\bi\s*want\s*to\s*(?:disappear|sleep\s*forever)\b",
        # Ambiguous slang (treat as LOW unless combined with 1st-person or nearby intent)
        r"\bkms\b",
        r"\b(?:unalive|self[-\s]*delete)\b",
    ],
}

# ============================================================================
# VALIDATION RESULTS
# ============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CRISIS = "crisis"

@dataclass
class GroundingValidation:
    result: ValidationResult
    grounding_score: float
    final_response: str
    reasoning: str = ""

# ============================================================================
# QUERY VALIDATOR WITH CRISIS DETECTION
# ============================================================================

class QueryValidator:
    """Query validator with crisis/self-harm detection"""
    
    @staticmethod
    def check_crisis_indicators(query: str) -> Tuple[bool, str]:
        """
        Check for crisis/self-harm indicators in query
        Returns (is_crisis, severity_level)
        """
        query_lower = query.lower()
        
        # Check HIGH severity patterns (immediate crisis)
        for pattern in CRISIS_REGEX["HIGH"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.critical(f"HIGH severity crisis detected: '{query[:50]}...'")
                return True, "HIGH"
        
        # Check MEDIUM severity patterns (planning/methods)
        medium_matches = 0
        for pattern in CRISIS_REGEX["MEDIUM"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                medium_matches += 1
        
        # Two or more medium patterns suggest crisis
        if medium_matches >= 2:
            logger.critical(f"MEDIUM severity crisis detected (multiple patterns): '{query[:50]}...'")
            return True, "MEDIUM"
        
        # Check LOW severity patterns
        low_matches = 0
        for pattern in CRISIS_REGEX["LOW"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                low_matches += 1
        
        # If we have 1 medium + 1 low, or 2+ low patterns, flag as potential crisis
        if (medium_matches >= 1 and low_matches >= 1) or low_matches >= 2:
            logger.warning(f"Potential crisis detected (combined patterns): '{query[:50]}...'")
            return True, "LOW"
        
        return False, None
    
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate query for crisis content
        Returns (is_blocked, message)
        """
        # Check for crisis/self-harm indicators
        is_crisis, severity = QueryValidator.check_crisis_indicators(query)
        
        if is_crisis:
            logger.critical(f"CRISIS QUERY BLOCKED - Severity: {severity}")
            return True, CRISIS_RESPONSE_MESSAGE
        
        # Query is OK to process
        return False, None

# ============================================================================
# GROUNDING VALIDATOR
# ============================================================================

class GroundingValidator:
    """
    Semantic grounding validator to ensure responses match context
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
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
        Validate response grounding against context
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
        
        # Skip validation for standard fallback messages
        response_lower = response.lower()
        if response_lower.startswith(("i'm sorry", "i don't have", "i cannot", "i'm really concerned")):
            return GroundingValidation(
                result=ValidationResult.APPROVED,
                grounding_score=1.0,
                final_response=response,
                reasoning="Standard fallback/crisis message"
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
            
            # Check against threshold
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