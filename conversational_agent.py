# conversational_agent.py - Simplified Response Orchestrator (updated for streaming)

import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, AsyncGenerator
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
    # final_response can now be a str or an AsyncGenerator for streaming
    final_response: str | AsyncGenerator[str, None] = ""
    strategy_used: ResponseStrategy = ResponseStrategy.GENERATED
    context_used: str = ""
    grounding_score: float = 0.0 # Will be final score for non-streaming, or initial for streaming
    latency_ms: int = 0 # Will be latency up to response generation start for streaming
    was_validated: bool = False
    validation_result: str = "" # Will be final result for non-streaming, or 'pending' for streaming
    cache_hit: bool = False
    # New fields to store the actual final validation results after a stream completes
    # These will be updated by the post-stream processor.
    _final_grounding_score: float = field(init=False, default=0.0)
    _final_validation_result: str = field(init=False, default="pending")
    _full_generated_response_text: Optional[str] = field(init=False, default=None)

    def set_post_stream_validation_results(self, score: float, result: str, full_text: str):
        """Sets the validation results after streaming has completed."""
        self._final_grounding_score = score
        self._final_validation_result = result
        self._full_generated_response_text = full_text
        # Update the main fields for consistency if decision object is checked later
        self.grounding_score = score
        self.validation_result = result


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
    2. Generates response using Claude (now with streaming)
    3. Validates grounding (after full response is available)
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
        Main orchestration pipeline, now handles streaming LLM responses.
        Returns a ResponseDecision where final_response can be a string or an AsyncGenerator.
        Grounding validation happens 'offline' (after stream completes).
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
                        latency_ms=latency,
                        was_validated=True, # Cached responses are considered validated
                        validation_result="approved",
                        _full_generated_response_text=cached_response # For consistency
                    )
            
            # Step 2: Retrieve context from RAG
            logger.info(f"[Request #{self.total_requests}] Retrieving context from RAG...")
            rag_start = time.time()
            
            from rag import retrieve_and_format_context
            context = retrieve_and_format_context(query)
            
            rag_latency = int((time.time() - rag_start) * 1000)
            logger.info(f"[Request #{self.total_requests}] RAG retrieval completed in {rag_latency}ms - retrieved {len(context)} chars")
            
            # If no significant context, immediately fallback without calling LLM
            if not context or len(context.strip()) < 50:
                logger.info(f"[Request #{self.total_requests}] No sufficient context retrieved - falling back early.")
                self.fallback_count += 1
                total_latency = int((time.time() - start_time) * 1000)
                return ResponseDecision(
                    final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                    strategy_used=ResponseStrategy.FALLBACK,
                    context_used="",
                    grounding_score=0.0,
                    latency_ms=total_latency,
                    was_validated=True, # Considered validated as it's a direct fallback
                    validation_result="no_context",
                    _full_generated_response_text=NO_CONTEXT_FALLBACK_MESSAGE
                )

            # Step 3: Generate response using Claude (now streaming)
            logger.info(f"[Request #{self.total_requests}] Requesting streaming response from Claude...")
            
            from llm_client import call_claude
            # call_claude now returns an AsyncGenerator
            llm_response_generator = call_claude(query, context, conversation_history)
            
            # Capture latency up to the point of initiating the LLM stream.
            latency_up_to_llm_init = int((time.time() - start_time) * 1000)

            # Create the ResponseDecision object here, with initial placeholders for post-stream results
            decision = ResponseDecision(
                final_response=None, # Will be set to the wrapped generator
                strategy_used=ResponseStrategy.GENERATED,
                context_used=context,
                grounding_score=0.0,
                latency_ms=latency_up_to_llm_init,
                was_validated=False, # Will be validated post-stream
                validation_result="pending", # Will be known post-stream
                cache_hit=False
            )

            # Define a wrapper async generator to perform grounding and caching *after* the stream completes
            async def _post_stream_processor(response_gen: AsyncGenerator[str, None], current_context: str, decision_obj: ResponseDecision) -> AsyncGenerator[str, None]:
                full_generated_response_buffer = []
                
                try:
                    async for chunk in response_gen:
                        full_generated_response_buffer.append(chunk)
                        yield chunk # Yield chunks to the UI as they arrive
                    
                    full_response_text = "".join(full_generated_response_buffer).strip()
                    logger.info(f"[Request #{self.total_requests}] LLM stream completed. Total chars: {len(full_response_text)}")

                    # Step 4: Validate grounding (after full response is available)
                    if current_context and len(current_context.strip()) > 50:
                        logger.info(f"[Request #{self.total_requests}] Validating response grounding (post-stream)...")
                        val_start = time.time()
                        
                        from guard import grounding_guard
                        validation = grounding_guard.validate_response(full_response_text, current_context)
                        
                        val_latency = int((time.time() - val_start) * 1000)
                        
                        # Update the decision object with the actual post-stream validation results
                        decision_obj.set_post_stream_validation_results(
                            validation.grounding_score,
                            validation.result.value,
                            validation.final_response # This is the potentially-guarded text
                        )
                        
                        if validation.result.value == "approved":
                            logger.info(f"[Request #{self.total_requests}] Response APPROVED with score {validation.grounding_score:.3f} in {val_latency}ms (post-stream)")
                            # Cache approved responses
                            if self.cache and cache_key:
                                self.cache.put(cache_key, full_response_text)
                        else:
                            logger.warning(f"[Request #{self.total_requests}] Response REJECTED with score {validation.grounding_score:.3f} - applying guard fallback (post-stream)")
                            self.fallback_count += 1
                            # If rejected, the full_response_text should be the guard's fallback message.
                            # We need to yield this fallback if the original stream was rejected.
                            # This means replacing the entire content, which is difficult after streaming.
                            # The compromise: The UI has streamed the content. The 'official' record in history
                            # (via conversation_manager) will be the fallback. The UI's display will be what streamed.
                            # This is a known challenge with post-stream guarding.
                            # For simplicity, we assume the UI will use the `_full_generated_response_text` for history.
                    else:
                        logger.info(f"[Request #{self.total_requests}] No context for post-stream validation (should have been caught earlier).")
                        decision_obj.set_post_stream_validation_results(
                            0.0,
                            "no_context",
                            NO_CONTEXT_FALLBACK_MESSAGE
                        )
                        self.fallback_count += 1

                except Exception as e:
                    logger.error(f"[Request #{self.total_requests}] Error during post-stream processing: {e}", exc_info=True)
                    decision_obj.set_post_stream_validation_results(
                        0.0, "error_post_stream", "An error occurred during response processing."
                    )
                    # If an error occurs during post-processing, yield a safe message at the end
                    yield "An error occurred during processing the response."
                finally:
                    # Log total request time once post-stream processing is done
                    total_time = time.time() - start_time
                    logger.info(f"[Request #{self.total_requests}] END Full request (including streaming & post-processing) completed in {total_time:.2f}s")
            
            # Set the final_response of the decision to the wrapped generator
            decision.final_response = _post_stream_processor(llm_response_generator, context, decision)
            return decision
            
        except Exception as e:
            logger.error(f"[Request #{self.total_requests}] Orchestration failed: {e}", exc_info=True)
            
            total_latency = int((time.time() - start_time) * 1000)
            
            # For errors, yield a single fallback message
            async def error_generator():
                yield NO_CONTEXT_FALLBACK_MESSAGE
            
            return ResponseDecision(
                final_response=error_generator(), # Return an async generator with fallback
                strategy_used=ResponseStrategy.ERROR,
                latency_ms=total_latency,
                validation_result="error",
                _full_generated_response_text=NO_CONTEXT_FALLBACK_MESSAGE
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