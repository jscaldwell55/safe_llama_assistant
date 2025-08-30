# conversational_agent.py - Orchestrator with Medical NER and Follow-up Continuity

import logging
import time
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from config import (
    ENABLE_RESPONSE_CACHE,
    MAX_CACHE_SIZE,
    NO_CONTEXT_FALLBACK_MESSAGE,
    LOG_SLOW_REQUESTS_THRESHOLD_MS,
    USE_TOP_SCORE_FOR_QUALITY,
    MIN_TOP_SCORE,
)

logger = logging.getLogger(__name__)

# ============================================================================
# RESPONSE STRATEGIES
# ============================================================================

class ResponseStrategy(Enum):
    GENERATED = "generated"
    CACHED = "cached"
    FALLBACK = "fallback"
    CRISIS = "crisis"
    ERROR = "error"

@dataclass
class ResponseDecision:
    """Response orchestration decision"""
    final_response: str = ""
    strategy_used: ResponseStrategy = ResponseStrategy.GENERATED
    context_used: str = ""
    grounding_score: float = 0.0
    latency_ms: int = 0
    was_validated: bool = False
    validation_result: str = ""
    cache_hit: bool = False
    crisis_detected: bool = False
    entities_found: List[Dict] = field(default_factory=list)

# ============================================================================
# RESPONSE CACHE
# ============================================================================

class ResponseCache:
    """Simple cache for validated responses"""

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_key(self, query: str) -> str:
        """Generate cache key from query"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache HIT for key: {key[:8]}...")
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: str):
        """Store validated response in cache"""
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value
        logger.debug(f"Cached response for key: {key[:8]}...")

    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Response cache cleared")

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class EnhancedOrchestrator:
    """
    Orchestrator with medical NER and robust follow-up handling.
    - Independently loads MedicalEntityRecognizer
    - Resolves follow-ups into explicit queries
    - Uses the SAME resolved query for retrieval, scoring, and formatting
    """

    def __init__(self):
        self.cache = ResponseCache() if ENABLE_RESPONSE_CACHE else None
        self.total_requests = 0
        self.fallback_count = 0
        self.crisis_count = 0

        self._init_features()
        logger.info("EnhancedOrchestrator initialized (Medical NER enabled; no conversation_flow/context_enhancer)")

    def _init_features(self):
        """Initialize optional features safely (only Medical NER is loaded)."""
        # Medical NER (load independently so missing optional modules don't disable it)
        try:
            from medical_entity_recognizer import get_medical_recognizer
            self.medical_recognizer = get_medical_recognizer()
            logger.info("Medical Entity Recognizer loaded")
        except Exception as e:
            logger.warning(f"Medical Entity Recognizer unavailable: {e}")
            self.medical_recognizer = None

    async def orchestrate_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> ResponseDecision:
        """
        End-to-end orchestration:
        1) Crisis check
        2) Resolve follow-up into explicit query (resolved_query)
        3) Entities & intent from resolved_query
        4) Cache check keyed by resolved_query (+ primary_drug if present)
        5) Retrieve/score using resolved_query
        6) Format context with conversation history
        7) Generate with Claude (original query + resolved context)
        8) Validate grounding
        """
        start_time = time.time()
        self.total_requests += 1
        logger.info(f"[Request #{self.total_requests}] Processing: '{query[:80]}...'")

        try:
            # ---------------------------------------------------------------
            # 1) Crisis check (fast-fail)
            # ---------------------------------------------------------------
            from guard import QueryValidator
            is_crisis, crisis_message = QueryValidator.validate_query(query)
            if is_crisis:
                self.crisis_count += 1
                latency = int((time.time() - start_time) * 1000)
                logger.critical(f"[Request #{self.total_requests}] CRISIS DETECTED")
                return ResponseDecision(
                    final_response=crisis_message,
                    strategy_used=ResponseStrategy.CRISIS,
                    latency_ms=latency,
                    validation_result="crisis_detected",
                    crisis_detected=True
                )

            # ---------------------------------------------------------------
            # 2) Resolve follow-ups consistently
            # ---------------------------------------------------------------
            from rag import resolve_followup
            resolved_query = resolve_followup(query, conversation_history)
            if resolved_query != query:
                logger.info(f"[Request #{self.total_requests}] Follow-up resolved → {resolved_query!r}")

            # ---------------------------------------------------------------
            # 3) Entities & intent (from resolved_query so cache/retrieval align)
            # ---------------------------------------------------------------
            entities = []
            query_intent = {}
            if self.medical_recognizer:
                try:
                    entities = self.medical_recognizer.extract_entities(resolved_query)
                    query_intent = self.medical_recognizer.analyze_query_intent(resolved_query, entities)
                    logger.info(f"[Request #{self.total_requests}] Entities detected: {len(entities)}")
                except Exception as e:
                    logger.warning(f"Entity analysis failed: {e}")

            # ---------------------------------------------------------------
            # 4) Cache (keyed on resolved_query + primary_drug to improve hits)
            # ---------------------------------------------------------------
            cache_key = None
            if self.cache:
                suffix = (query_intent.get("primary_drug") or "").strip()
                cache_key = self.cache.get_key(resolved_query + ("|" + suffix if suffix else ""))
                cached = self.cache.get(cache_key)
                if cached:
                    latency = int((time.time() - start_time) * 1000)
                    logger.info(f"[Request #{self.total_requests}] Cache HIT in {latency}ms")
                    return ResponseDecision(
                        final_response=cached,
                        strategy_used=ResponseStrategy.CACHED,
                        cache_hit=True,
                        latency_ms=latency,
                        entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
                    )

            # ---------------------------------------------------------------
            # 5) Retrieval & quality scoring (ALWAYS use resolved_query)
            # ---------------------------------------------------------------
            from rag import retrieve_and_format_context, get_rag_system
            rag_system = get_rag_system()

            # Raw results for quality signals
            results = rag_system.retrieve(resolved_query)
            top_score = results[0]["score"] if results else 0.0

            if results and USE_TOP_SCORE_FOR_QUALITY and top_score < MIN_TOP_SCORE:
                logger.warning(
                    f"[Request #{self.total_requests}] Poor retrieval: top score {top_score:.3f} < {MIN_TOP_SCORE}"
                )
                self.fallback_count += 1
                return ResponseDecision(
                    final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                    strategy_used=ResponseStrategy.FALLBACK,
                    latency_ms=int((time.time() - start_time) * 1000),
                    validation_result="poor_retrieval",
                    entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
                )

            # ---------------------------------------------------------------
            # 6) Context formatting (pass conversation_history; function re-resolves safely)
            # ---------------------------------------------------------------
            context = retrieve_and_format_context(resolved_query, conversation_history=conversation_history)
            if not context or len(context) < 50:
                logger.info(f"[Request #{self.total_requests}] No/insufficient context → fallback")
                self.fallback_count += 1
                return ResponseDecision(
                    final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                    strategy_used=ResponseStrategy.FALLBACK,
                    latency_ms=int((time.time() - start_time) * 1000),
                    validation_result="no_context",
                    entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
                )

            # ---------------------------------------------------------------
            # 7) Generate with Claude (use original user phrasing; resolved context)
            # ---------------------------------------------------------------
            from llm_client import call_claude
            response = await call_claude(query, context, conversation_history)

            # ---------------------------------------------------------------
            # 8) Grounding validation
            # ---------------------------------------------------------------
            from guard import grounding_validator
            validation = grounding_validator.validate_response(response, context)

            if validation.result.value != "approved":
                logger.warning(f"[Request #{self.total_requests}] Validation rejected: {validation.reasoning}")
                self.fallback_count += 1
                final = validation.final_response
                score = validation.grounding_score
                validated = True
                vresult = "rejected"
            else:
                final = response
                score = validation.grounding_score
                validated = True
                vresult = "approved"
                # Cache only approved responses
                if self.cache and cache_key:
                    self.cache.put(cache_key, final)

            total_latency = int((time.time() - start_time) * 1000)
            if total_latency > LOG_SLOW_REQUESTS_THRESHOLD_MS:
                logger.warning(f"[PERF] Slow request #{self.total_requests}: {total_latency}ms")

            return ResponseDecision(
                final_response=final,
                strategy_used=ResponseStrategy.GENERATED,
                context_used=context,
                grounding_score=score,
                latency_ms=total_latency,
                was_validated=validated,
                validation_result=vresult,
                entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
            )

        except Exception as e:
            logger.error(f"[Request #{self.total_requests}] Orchestration failed: {e}", exc_info=True)
            return ResponseDecision(
                final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                strategy_used=ResponseStrategy.ERROR,
                latency_ms=int((time.time() - start_time) * 1000),
                validation_result="error"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        stats = {
            "total_requests": self.total_requests,
            "fallback_count": self.fallback_count,
            "crisis_count": self.crisis_count,
        }
        if self.cache:
            stats.update({
                "cache_size": len(self.cache.cache),
                "cache_hits": self.cache.hits,
                "cache_misses": self.cache.misses,
                "cache_hit_rate": self.cache.get_hit_rate(),
            })
        return stats

# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_orchestrator_instance: Optional[EnhancedOrchestrator] = None

def get_orchestrator() -> EnhancedOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = EnhancedOrchestrator()
        logger.info("Created EnhancedOrchestrator singleton")
    return _orchestrator_instance

def reset_orchestrator():
    """Reset orchestrator state"""
    orchestrator = get_orchestrator()
    if orchestrator.cache:
        orchestrator.cache.clear()
    orchestrator.total_requests = 0
    orchestrator.fallback_count = 0
    orchestrator.crisis_count = 0
    logger.info("Orchestrator state reset")
