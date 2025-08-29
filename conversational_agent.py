# conversational_agent.py - Enhanced Orchestrator with Medical NER and Conversation Flows

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
    CRISIS = "crisis"
    CLARIFICATION = "clarification"  # For clarification requests
    ERROR = "error"

@dataclass
class ResponseDecision:
    """Enhanced response orchestration decision"""
    final_response: str = ""
    strategy_used: ResponseStrategy = ResponseStrategy.GENERATED
    context_used: str = ""
    grounding_score: float = 0.0
    latency_ms: int = 0
    was_validated: bool = False
    validation_result: str = ""
    cache_hit: bool = False
    crisis_detected: bool = False
    entities_found: List[Dict] = field(default_factory=list)  # Medical entities
    clarification_needed: bool = False  # Clarification flag

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
# ENHANCED ORCHESTRATOR
# ============================================================================

class EnhancedOrchestrator:
    """
    Enhanced orchestrator with medical NER, conversation flows, and confidence scoring
    """
    
    def __init__(self):
        self.cache = ResponseCache() if ENABLE_RESPONSE_CACHE else None
        self.total_requests = 0
        self.fallback_count = 0
        self.crisis_count = 0
        self.clarification_count = 0
        
        # Initialize new components
        self._init_enhanced_features()
        
        logger.info(f"EnhancedOrchestrator initialized with medical NER and conversation flows")
    
    def _init_enhanced_features(self):
        """Initialize enhanced features"""
        try:
            from medical_entity_recognizer import get_medical_recognizer
            from conversation_flow import get_flow_manager
            from context_enhancer import get_query_enhancer
            
            self.medical_recognizer = get_medical_recognizer()
            self.flow_manager = get_flow_manager()
            self.query_enhancer = get_query_enhancer()
            self.enhanced_features = True
            
            logger.info("Enhanced features loaded: Medical NER, Conversation Flows, and Query Enhancement")
        except ImportError as e:
            logger.warning(f"Enhanced features not available: {e}")
            self.medical_recognizer = None
            self.flow_manager = None
            self.query_enhancer = None
            self.enhanced_features = False
    
    async def orchestrate_response(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> ResponseDecision:
        """
        Enhanced orchestration with medical understanding
        """
        start_time = time.time()
        self.total_requests += 1
        
        logger.info(f"[Request #{self.total_requests}] Processing: '{query[:50]}...'")
        
        try:
            # Step 1: CRITICAL - Check for crisis/self-harm first
            from guard import QueryValidator
            is_crisis, crisis_message = QueryValidator.validate_query(query)
            
            if is_crisis:
                self.crisis_count += 1
                latency = int((time.time() - start_time) * 1000)
                
                logger.critical(f"[Request #{self.total_requests}] CRISIS DETECTED - Returning crisis response")
                
                return ResponseDecision(
                    final_response=crisis_message,
                    strategy_used=ResponseStrategy.CRISIS,
                    latency_ms=latency,
                    validation_result="crisis_detected",
                    crisis_detected=True
                )
            
            # Step 1.5: Enhance query with conversation context if needed
            enhanced_query = query
            was_enhanced = False
            
            if self.query_enhancer and conversation_history:
                enhanced_query, was_enhanced = self.query_enhancer.enhance_query(query, conversation_history)
                if was_enhanced:
                    logger.info(f"[Request #{self.total_requests}] Query enhanced with context")
            
            # Step 2: Extract medical entities and analyze intent
            entities = []
            query_intent = {}
            
            if self.medical_recognizer:
                # Use enhanced query for entity extraction
                entities = self.medical_recognizer.extract_entities(enhanced_query)
                query_intent = self.medical_recognizer.analyze_query_intent(enhanced_query, entities)
                
                logger.info(f"[Request #{self.total_requests}] Found {len(entities)} medical entities")
                for entity in entities[:3]:  # Log first 3 entities
                    logger.debug(f"  - {entity.entity_type.value}: {entity.text}")
            
            # Step 3: Check cache (with entity-aware key)
            cache_key = None
            if self.cache:
                # Include primary drug in cache key if present
                cache_suffix = query_intent.get("primary_drug", "") or ""
                cache_key = self.cache.get_key(query + cache_suffix)
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    latency = int((time.time() - start_time) * 1000)
                    logger.info(f"[Request #{self.total_requests}] Cache hit - returning in {latency}ms")
                    return ResponseDecision(
                        final_response=cached_response,
                        strategy_used=ResponseStrategy.CACHED,
                        cache_hit=True,
                        latency_ms=latency,
                        entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
                    )
            
            # Step 4: Retrieve context from RAG
            logger.info(f"[Request #{self.total_requests}] Retrieving context...")
            rag_start = time.time()

            from rag import retrieve_and_format_context, get_rag_system
            rag_system = get_rag_system()

            # Pass conversation history for better follow-up handling
            context = retrieve_and_format_context(query, conversation_history=conversation_history)

            # Get raw retrieval results for scoring (using same query)
            results = rag_system.retrieve(query)
            retrieval_scores = [r["score"] for r in results] if results else []
            
            # Step 5: Check if clarification is needed
            if self.flow_manager and results:
                clarifications = self.flow_manager.analyze_ambiguity(query, query_intent)
                top_score = retrieval_scores[0] if retrieval_scores else 0
                
                if self.flow_manager.should_ask_clarification(clarifications, top_score):
                    clarification_response = self.flow_manager.format_clarification_response(clarifications)
                    
                    if clarification_response:
                        self.clarification_count += 1
                        logger.info(f"[Request #{self.total_requests}] Requesting clarification")
                        
                        return ResponseDecision(
                            final_response=clarification_response,
                            strategy_used=ResponseStrategy.CLARIFICATION,
                            latency_ms=int((time.time() - start_time) * 1000),
                            clarification_needed=True,
                            entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
                        )
            
            # Step 6: Check retrieval quality
            if results:
                if USE_TOP_SCORE_FOR_QUALITY:
                    top_score = retrieval_scores[0] if retrieval_scores else 0
                    if top_score < MIN_TOP_SCORE:
                        logger.warning(f"[Request #{self.total_requests}] Poor retrieval quality: top score {top_score:.3f} < {MIN_TOP_SCORE}")
                        self.fallback_count += 1
                        
                        return ResponseDecision(
                            final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                            strategy_used=ResponseStrategy.FALLBACK,
                            latency_ms=int((time.time() - start_time) * 1000),
                            validation_result="poor_retrieval",
                            entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
                        )
            
            # Format context using enhanced query
            context = retrieve_and_format_context(enhanced_query)
            
            rag_latency = int((time.time() - rag_start) * 1000)
            logger.info(f"[Request #{self.total_requests}] RAG completed in {rag_latency}ms - {len(context)} chars")
            
            # Step 7: Generate response
            if not context or len(context) < 50:
                logger.info(f"[Request #{self.total_requests}] No context - using fallback")
                self.fallback_count += 1
                
                return ResponseDecision(
                    final_response=NO_CONTEXT_FALLBACK_MESSAGE,
                    strategy_used=ResponseStrategy.FALLBACK,
                    latency_ms=int((time.time() - start_time) * 1000),
                    validation_result="no_context",
                    entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
                )
            
            logger.info(f"[Request #{self.total_requests}] Generating response...")
            gen_start = time.time()
            
            from llm_client import call_claude
            response = await call_claude(query, context, conversation_history)
            
            gen_latency = int((time.time() - gen_start) * 1000)
            logger.info(f"[Request #{self.total_requests}] Generation completed in {gen_latency}ms")
            
            # Step 8: Validate grounding
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
                validation_result=validation_result,
                entities_found=[{"text": e.text, "type": e.entity_type.value} for e in entities]
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
            "crisis_count": self.crisis_count,
            "clarification_count": self.clarification_count,
            "fallback_rate": self.fallback_count / self.total_requests if self.total_requests > 0 else 0,
            "crisis_rate": self.crisis_count / self.total_requests if self.total_requests > 0 else 0,
            "clarification_rate": self.clarification_count / self.total_requests if self.total_requests > 0 else 0,
            "enhanced_features": self.enhanced_features
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
    orchestrator.clarification_count = 0
    logger.info("Orchestrator state reset")