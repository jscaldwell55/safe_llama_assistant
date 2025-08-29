# conversational_agent.py - Response Orchestrator with Query Pre-screening

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
    PERSONAL_MEDICAL_ADVICE_MESSAGE,
    LOG_SLOW_REQUESTS_THRESHOLD_MS,
    MIN_RETRIEVAL_SCORE,
    USE_TOP_SCORE_FOR_QUALITY,
    MIN_TOP_SCORE
)

logger = logging.getLogger(__name__)

# ============================================================================
# RESPONSE STRATEGIES
# ============================================================================

class ResponseStrategy(Enum):
    GENERATED = "generated"
    CACHED = "cached"
    FALLBACK = "fallback"
    BLOCKED = "blocked"
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
    blocked_reason: str = ""

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
# ORCHESTRATOR WITH PRE-SCREENING
# ============================================================================

class SafeOrchestrator:
    """
    Production orchestrator with query pre-screening and safe thresholds
    """
    
    def __init__(self):
        self.cache = ResponseCache() if ENABLE_RESPONSE_CACHE else None
        self.total_requests = 0
        self.fallback_count = 0
        self.blocked_count = 0
        
        logger.info(f"SafeOrchestrator initialized (cache: {ENABLE_RESPONSE_CACHE})")
    
    async def orchestrate_response(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> ResponseDecision:
        """
        Main orchestration with query pre-screening
        """
        start_time = time.time()
        self.total_requests += 1
        
        logger.info(f"[Request #{self.total_requests}] Processing: '{query[:50]}...'")
        
        try:
            # Step 1: Pre-screen query for emergencies and personal medical advice
            from guard import QueryValidator
            
            # Call validate_query as a static method directly on the class
            is_blocked, block_message = QueryValidator.validate_query(query)
            
            if is_blocked:
                self.blocked_count += 1
                latency = int((time.time() - start_time) * 1000)
                
                # Determine block reason
                if "emergency" in block_message.lower() or "911" in block_message:
                    block_reason = "emergency"
                    validation_result = "blocked_emergency"
                else:
                    block_reason = "personal_medical_advice"
                    validation_result = "blocked_personal_medical"
                
                logger.info(f"[Request #{self.total_requests}] Query blocked - {block_reason}")
                
                return ResponseDecision(
                    final_response=block_message,
                    strategy_used=ResponseStrategy.BLOCKED,
                    latency_ms=latency,
                    validation_result=validation_result,
                    blocked_reason=block_reason
                )
            
            # Step 2: Check cache
            cache_key = None
            if self.cache:
                cache_key = self.cache.get_key(query)
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    latency = int((time.time() - start_time) * 1000)
                    logger.info(f"[Request #{self.total_requests}] Cache hit - returning in {latency}ms")
                    return ResponseDecision(
                        final_response=cached_response,
                        strategy_used=ResponseStrategy.CACHED,
                        cache_hit=True,
                        latency_ms=latency
                    )
            
            # Step 3: Retrieve context from RAG
            logger.info(f"[Request #{self.total_requests}] Retrieving context...")
            rag_start = time.time()
            
            from rag import retrieve_and_format_context, get_rag_system
            rag_system = get_rag_system()
            
            # Get raw retrieval results to check quality
            results = rag_system.retrieve(query)
            
            # Check retrieval quality
            if results:
                if USE_TOP_SCORE_FOR_QUALITY:
                    # Use the best chunk's score
                    top_score = results[0]["score"] if results else 0
                    if top_score < MIN_TOP_SCORE:
                        logger.warning(f"[Request #{self.total_requests}] Poor retrieval quality: top score {top_score:.3f} < {MIN_TOP_SCORE}")
                        self.fallback_count += 1
                        return ResponseDecision(
                            final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                            strategy_used=ResponseStrategy.FALLBACK,
                            latency_ms=int((time.time() - start_time) * 1000),
                            validation_result="poor_retrieval"
                        )
                else:
                    # Use average score
                    avg_score = sum(r["score"] for r in results) / len(results)
                    if avg_score < MIN_RETRIEVAL_SCORE:
                        logger.warning(f"[Request #{self.total_requests}] Poor retrieval quality: {avg_score:.3f} < {MIN_RETRIEVAL_SCORE}")
                        self.fallback_count += 1
                        return ResponseDecision(
                            final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                            strategy_used=ResponseStrategy.FALLBACK,
                            latency_ms=int((time.time() - start_time) * 1000),
                            validation_result="poor_retrieval"
                        )
            
            # Format context
            context = retrieve_and_format_context(query)
            
            rag_latency = int((time.time() - rag_start) * 1000)
            logger.info(f"[Request #{self.total_requests}] RAG completed in {rag_latency}ms - {len(context)} chars")
            
            # Step 4: Generate response
            if not context or len(context) < 50:
                logger.info(f"[Request #{self.total_requests}] No context - using fallback")
                self.fallback_count += 1
                return ResponseDecision(
                    final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                    strategy_used=ResponseStrategy.FALLBACK,
                    latency_ms=int((time.time() - start_time) * 1000),
                    validation_result="no_context"
                )
            
            logger.info(f"[Request #{self.total_requests}] Generating response...")
            gen_start = time.time()
            
            from llm_client import call_claude
            response = await call_claude(query, context, conversation_history)
            
            gen_latency = int((time.time() - gen_start) * 1000)
            logger.info(f"[Request #{self.total_requests}] Generation completed in {gen_latency}ms")
            
            # Step 5: Validate grounding
            logger.info(f"[Request #{self.total_requests}] Validating grounding...")
            val_start = time.time()
            
            from guard import grounding_validator
            validation = grounding_validator.validate_response(response, context)
            
            val_latency = int((time.time() - val_start) * 1000)
            
            if validation.result.value == "approved":
                logger.info(f"[Request #{self.total_requests}] APPROVED: {validation.reasoning}")
                
                # Cache approved responses
                if self.cache and cache_key:
                    self.cache.put(cache_key, response)
                
                validation_result = "approved"
            else:
                logger.warning(f"[Request #{self.total_requests}] REJECTED: {validation.reasoning}")
                response = validation.final_response
                self.fallback_count += 1
                validation_result = "rejected"
            
            # Calculate total latency
            total_latency = int((time.time() - start_time) * 1000)
            
            if total_latency > LOG_SLOW_REQUESTS_THRESHOLD_MS:
                logger.warning(f"[PERF] Slow request #{self.total_requests}: {total_latency}ms")
            
            return ResponseDecision(
                final_response=response,
                strategy_used=ResponseStrategy.GENERATED,
                context_used=context,
                grounding_score=validation.grounding_score,
                latency_ms=total_latency,
                was_validated=True,
                validation_result=validation_result
            )
            
        except Exception as e:
            logger.error(f"[Request #{self.total_requests}] Failed: {e}", exc_info=True)
            
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
            "blocked_count": self.blocked_count,
            "fallback_rate": self.fallback_count / self.total_requests if self.total_requests > 0 else 0,
            "blocked_rate": self.blocked_count / self.total_requests if self.total_requests > 0 else 0
        }
        
        if self.cache:
            stats.update({
                "cache_size": len(self.cache.cache),
                "cache_hits": self.cache.hits,
                "cache_misses": self.cache.misses,
                "cache_hit_rate": self.cache.get_hit_rate()
            })
        
        return stats

# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_orchestrator_instance: Optional[SafeOrchestrator] = None

def get_orchestrator() -> SafeOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SafeOrchestrator()
        logger.info("Created SafeOrchestrator singleton")
    return _orchestrator_instance

def reset_orchestrator():
    """Reset orchestrator state"""
    orchestrator = get_orchestrator()
    if orchestrator.cache:
        orchestrator.cache.clear()
    orchestrator.total_requests = 0
    orchestrator.fallback_count = 0
    orchestrator.blocked_count = 0
    logger.info("Orchestrator state reset")