# conversational_agent.py - Simplified Response Orchestrator (Non-streaming)

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
    LOG_SLOW_REQUESTS_THRESHOLD_MS
)

logger = logging.getLogger(__name__)

# ============================================================================
# RESPONSE STRATEGIES
# ============================================================================

class ResponseStrategy(Enum):
    GENERATED = "generated"          # Response generated from context
    CACHED = "cached"                # Previously cached response
    FALLBACK = "fallback"            # No context available
    ERROR = "error"                  # Error occurred

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
            logger.debug(f"Cache HIT for key: {key[:8]}... (hit rate: {self.get_hit_rate():.1%})")
            return self.cache[key]
        self.misses += 1
        logger.debug(f"Cache MISS for key: {key[:8]}...")
        return None
    
    def put(self, key: str, value: str):
        """Store validated response in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            logger.debug(f"Cache full - evicted oldest entry")
        
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
# SIMPLIFIED ORCHESTRATOR
# ============================================================================

class SimpleOrchestrator:
    """
    Simplified orchestrator that:
    1. Retrieves context from RAG
    2. Generates response using Claude
    3. Validates grounding
    4. Caches approved responses
    """
    
    def __init__(self):
        self.cache = ResponseCache() if ENABLE_RESPONSE_CACHE else None
        self.total_requests = 0
        self.fallback_count = 0
        
        logger.info(f"SimpleOrchestrator initialized (cache: {ENABLE_RESPONSE_CACHE})")
    
    async def orchestrate_response(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> ResponseDecision:
        """
        Main orchestration pipeline - NON-STREAMING VERSION
        """
        start_time = time.time()
        self.total_requests += 1
        
        logger.info(f"[Request #{self.total_requests}] Processing query: '{query[:50]}...'")
        
        try:
            # Step 1: Check cache
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
            
            # Step 2: Retrieve context from RAG
            logger.info(f"[Request #{self.total_requests}] Retrieving context from RAG...")
            rag_start = time.time()
            
            from rag import retrieve_and_format_context
            context = retrieve_and_format_context(query)
            
            rag_latency = int((time.time() - rag_start) * 1000)
            logger.info(f"[Request #{self.total_requests}] RAG retrieval completed in {rag_latency}ms - retrieved {len(context)} chars")
            
            # Step 3: Generate response using Claude (NON-STREAMING)
            logger.info(f"[Request #{self.total_requests}] Generating response with Claude...")
            gen_start = time.time()
            
            from llm_client import call_claude
            response = await call_claude(query, context, conversation_history)
            
            gen_latency = int((time.time() - gen_start) * 1000)
            logger.info(f"[Request #{self.total_requests}] Claude generation completed in {gen_latency}ms")
            
            # Step 4: Validate grounding (if context exists)
            validation_result = "not_needed"
            grounding_score = 0.0
            was_validated = False
            
            if context and len(context) > 50:
                logger.info(f"[Request #{self.total_requests}] Validating response grounding...")
                val_start = time.time()
                
                from guard import grounding_guard
                validation = grounding_guard.validate_response(response, context)
                
                val_latency = int((time.time() - val_start) * 1000)
                was_validated = True
                grounding_score = validation.grounding_score
                
                if validation.result.value == "approved":
                    logger.info(f"[Request #{self.total_requests}] Response APPROVED with score {grounding_score:.3f} in {val_latency}ms")
                    validation_result = "approved"
                    
                    # Cache approved responses
                    if self.cache and cache_key:
                        self.cache.put(cache_key, response)
                else:
                    logger.warning(f"[Request #{self.total_requests}] Response REJECTED with score {grounding_score:.3f} - using fallback")
                    response = validation.final_response
                    validation_result = "rejected"
                    self.fallback_count += 1
            else:
                # No context case
                logger.info(f"[Request #{self.total_requests}] No context retrieved - using fallback")
                response = NO_CONTEXT_FALLBACK_MESSAGE
                self.fallback_count += 1
                validation_result = "no_context"
            
            # Calculate total latency
            total_latency = int((time.time() - start_time) * 1000)
            
            # Log performance warning if slow
            if total_latency > LOG_SLOW_REQUESTS_THRESHOLD_MS:
                logger.warning(f"[PERF] Slow request #{self.total_requests}: {total_latency}ms (RAG: {rag_latency}ms, Gen: {gen_latency}ms)")
            else:
                logger.info(f"[Request #{self.total_requests}] Completed in {total_latency}ms")
            
            # Determine strategy
            if not context:
                strategy = ResponseStrategy.FALLBACK
            else:
                strategy = ResponseStrategy.GENERATED
            
            return ResponseDecision(
                final_response=response,
                strategy_used=strategy,
                context_used=context,
                grounding_score=grounding_score,
                latency_ms=total_latency,
                was_validated=was_validated,
                validation_result=validation_result,
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"[Request #{self.total_requests}] Orchestration failed: {e}", exc_info=True)
            
            total_latency = int((time.time() - start_time) * 1000)
            
            return ResponseDecision(
                final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                strategy_used=ResponseStrategy.ERROR,
                latency_ms=total_latency,
                validation_result="error"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        stats = {
            "total_requests": self.total_requests,
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_count / self.total_requests if self.total_requests > 0 else 0
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

_orchestrator_instance: Optional[SimpleOrchestrator] = None

def get_orchestrator() -> SimpleOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SimpleOrchestrator()
        logger.info("Created SimpleOrchestrator singleton")
    return _orchestrator_instance

def reset_orchestrator():
    """Reset orchestrator state"""
    orchestrator = get_orchestrator()
    if orchestrator.cache:
        orchestrator.cache.clear()
    orchestrator.total_requests = 0
    orchestrator.fallback_count = 0
    logger.info("Orchestrator state reset")

# Legacy compatibility
def get_persona_conductor():
    """Legacy compatibility"""
    return get_orchestrator()