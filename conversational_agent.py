# conversational_agent.py - Fixed to Always Validate Responses
"""
Document-grounded agent with mandatory validation for all responses
"""

import logging
import re
import time
import hashlib
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# RESPONSE STRATEGIES
# ============================================================================

class ResponseStrategy(Enum):
    SYNTHESIZED = "synthesized"        # Response synthesized from documentation
    CONVERSATIONAL = "conversational"  # Simple conversational response
    CACHED = "cached"                  # Previously cached response
    BLOCKED = "blocked"                # Query blocked by safety
    ERROR = "error"                    # Error occurred

class ConversationMode(Enum):
    """Legacy compatibility"""
    GENERAL = "general"
    SESSION_END = "session_end"

@dataclass
class ConductorDecision:
    """Main decision structure for response orchestration"""
    final_response: str = ""
    requires_validation: bool = True  # ALWAYS TRUE NOW
    strategy_used: ResponseStrategy = ResponseStrategy.SYNTHESIZED
    context_used: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: int = 0
    was_blocked: bool = False
    grounding_score: float = 0.0

@dataclass 
class AgentDecision:
    """Legacy compatibility structure"""
    mode: ConversationMode = ConversationMode.GENERAL
    requires_generation: bool = True
    context_str: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# RESPONSE CACHE
# ============================================================================

class ResponseCache:
    """Simple cache for validated responses"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get_key(self, query: str, context: str = "") -> str:
        """Generate cache key from query and context"""
        combined = f"{query.lower().strip()}:{context[:100] if context else ''}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        if key in self.cache:
            logger.info(f"Cache hit for key: {key[:8]}...")
            return self.cache[key]
        return None
    
    def put(self, key: str, value: str):
        """Store validated response in cache"""
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = value
        logger.info(f"Cached validated response for key: {key[:8]}...")

# ============================================================================
# MAIN CONDUCTOR CLASS
# ============================================================================

class PersonaConductor:
    """
    Main orchestrator ensuring all responses are validated
    """
    
    def __init__(self):
        self._retriever = None
        self._guard = None
        self.cache = None
        
        # Load configuration
        try:
            from config import ENABLE_RESPONSE_CACHE, MAX_CACHE_SIZE
            if ENABLE_RESPONSE_CACHE:
                self.cache = ResponseCache(max_size=MAX_CACHE_SIZE)
        except ImportError:
            logger.warning("Could not load cache config")
        
        # Standard responses that don't need generation
        self.instant_responses = {
            "hello": "Hello! How can I help you with information about Journvax today?",
            "hi": "Hi there! What would you like to know about Journvax?",
            "hey": "Hey! How can I help you with Journvax information?",
            "thank you": "You're welcome! Is there anything else about Journvax I can help with?",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "goodbye": "Goodbye! Take care!",
            "bye": "Bye! Have a great day!",
        }
    
    def _get_retriever(self):
        """Lazy load RAG retriever"""
        if self._retriever is None:
            try:
                from rag import retrieve_and_format_context
                self._retriever = retrieve_and_format_context
            except ImportError:
                logger.error("Could not import RAG retriever")
        return self._retriever
    
    def _get_guard(self):
        """Lazy load guard"""
        if self._guard is None:
            try:
                from guard import hybrid_guard
                self._guard = hybrid_guard
            except ImportError:
                logger.error("Could not import guard")
        return self._guard
    
    async def generate_synthesized_response(self, query: str, context: str) -> str:
        """
        Generate response strictly grounded in context
        """
        from llm_client import call_huggingface_with_retry
        from config import BRIDGE_SYNTHESIZER_PARAMS
        from prompts import format_synthesizer_prompt
        
        # Use enhanced prompt with strict grounding
        prompt = format_synthesizer_prompt(query, context)
        
        try:
            response = await call_huggingface_with_retry(prompt, BRIDGE_SYNTHESIZER_PARAMS)
            
            if response.startswith("Error:"):
                return "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"
    
    async def orchestrate_response(self, query: str) -> ConductorDecision:
        """
        Main orchestration with MANDATORY validation for all responses
        """
        start_time = time.time()
        
        try:
            # Step 1: Pre-screen query for safety violations
            guard = self._get_guard()
            if guard:
                query_validation = await guard.validate_query(query)
                
                if query_validation is not None:
                    # Query is blocked - return with clear debug info
                    return ConductorDecision(
                        final_response=query_validation.final_response,
                        requires_validation=False,  # No need to validate a refusal
                        strategy_used=ResponseStrategy.BLOCKED,
                        was_blocked=True,
                        debug_info={
                            "blocked_reason": query_validation.reasoning,
                            "violation": query_validation.violation.value,
                            "validation": {
                                "result": "query_blocked",  # Clear that QUERY was blocked
                                "stage": "pre_screening",   # Where it was blocked
                                "response": "standard_refusal"  # What was returned
                            }
                        }
                    )
            
            # Step 2: Check cache if enabled
            cache_key = None
            if self.cache:
                cache_key = self.cache.get_key(query)
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    return ConductorDecision(
                        final_response=cached_response,
                        requires_validation=True,  # ALWAYS TRUE (even for cached)
                        strategy_used=ResponseStrategy.CACHED,
                        total_latency_ms=int((time.time() - start_time) * 1000),
                        debug_info={
                            "cache_hit": True,
                            "validation": {"result": "cached_approved"}
                        }
                    )
            
            # Step 3: Check for instant responses (greetings)
            query_lower = query.lower().strip()
            if query_lower in self.instant_responses:
                response = self.instant_responses[query_lower]
                
                # Even instant responses get validated
                if guard:
                    validation_result = await guard.validate_response(
                        response=response,
                        context="",
                        query=query
                    )
                    
                    if validation_result.result.value != "approved":
                        response = validation_result.final_response
                
                if self.cache and cache_key:
                    self.cache.put(cache_key, response)
                
                return ConductorDecision(
                    final_response=response,
                    requires_validation=True,  # ALWAYS TRUE
                    strategy_used=ResponseStrategy.CONVERSATIONAL,
                    total_latency_ms=int((time.time() - start_time) * 1000),
                    debug_info={
                        "instant_response": True,
                        "validation": {"result": "instant_validated"}
                    }
                )
            
            # Step 4: Retrieve context
            context = ""
            rag_time = 0
            rag_start = time.time()
            try:
                retriever = self._get_retriever()
                if retriever:
                    context = retriever(query)
                    logger.info(f"Retrieved context: {len(context)} chars")
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
                context = ""
            rag_time = int((time.time() - rag_start) * 1000)
            
            # Step 5: Generate response
            gen_start = time.time()
            response = await self.generate_synthesized_response(query, context)
            gen_time = int((time.time() - gen_start) * 1000)
            
            # Step 6: MANDATORY validation of generated response
            validation_result = None
            if guard:
                validation_result = await guard.validate_response(
                    response=response,
                    context=context,
                    query=query,
                    strategy_used=ResponseStrategy.SYNTHESIZED.value
                )
                
                # Handle validation result
                if validation_result.result.value != "approved":
                    logger.warning(f"Response rejected/corrected: {validation_result.reasoning}")
                    response = validation_result.final_response
                    # Don't cache rejected/corrected responses
                else:
                    # Cache only validated, approved responses
                    if self.cache and cache_key:
                        self.cache.put(cache_key, response)
            else:
                # If guard is not available, use standard refusal
                logger.error("Guard not available - using standard refusal")
                response = "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"
            
            # Calculate total time
            total_time = int((time.time() - start_time) * 1000)
            
            # Build comprehensive debug info
            debug_info = {
                "timing": {
                    "rag_ms": rag_time,
                    "generation_ms": gen_time,  
                    "total_ms": total_time
                },
                "context_length": len(context),
                "used_context": bool(context)
            }
            
            # Include validation info
            if validation_result:
                debug_info["validation"] = {
                    "result": validation_result.result.value,
                    "grounding_score": getattr(validation_result, 'grounding_score', 0.0),
                    "violation": validation_result.violation.value if hasattr(validation_result, 'violation') else None,
                    "was_corrected": validation_result.result.value != "approved",
                    "confidence": getattr(validation_result, 'confidence', 0.0),
                    "reasoning": validation_result.reasoning
                }
            else:
                debug_info["validation"] = {
                    "result": "no_guard",
                    "reason": "Guard not available - defaulted to refusal"
                }
            
            return ConductorDecision(
                final_response=response,
                requires_validation=True,  # ALWAYS TRUE
                strategy_used=ResponseStrategy.SYNTHESIZED,
                context_used=context,
                total_latency_ms=total_time,
                grounding_score=validation_result.grounding_score if validation_result else 0.0,
                debug_info=debug_info
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            
            return ConductorDecision(
                final_response="I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?",
                requires_validation=True,  # ALWAYS TRUE
                strategy_used=ResponseStrategy.ERROR,
                total_latency_ms=int((time.time() - start_time) * 1000),
                debug_info={
                    "error": str(e),
                    "validation": {"result": "error"}
                }
            )
    
    def reset_conversation(self):
        """Reset conversation state"""
        logger.info("Conversation state reset")

# ============================================================================
# LEGACY COMPATIBILITY CLASSES
# ============================================================================

class ConversationalAgent:
    """Legacy adapter for backward compatibility"""
    
    def __init__(self):
        self.conductor = get_persona_conductor()
    
    def process_query(self, query: str) -> AgentDecision:
        """Legacy synchronous interface"""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.conductor.orchestrate_response(query)
                    )
                    decision = future.result()
            else:
                decision = loop.run_until_complete(
                    self.conductor.orchestrate_response(query)
                )
            
            return AgentDecision(
                mode=ConversationMode.GENERAL,
                requires_generation=True,
                context_str=decision.context_used,
                debug_info=decision.debug_info
            )
            
        except Exception as e:
            logger.error(f"Legacy adapter error: {e}")
            return AgentDecision(
                mode=ConversationMode.GENERAL,
                requires_generation=True,
                context_str="",
                debug_info={"error": str(e)}
            )

# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_conductor_instance: Optional[PersonaConductor] = None

def get_persona_conductor() -> PersonaConductor:
    """Get singleton PersonaConductor instance"""
    global _conductor_instance
    if _conductor_instance is None:
        _conductor_instance = PersonaConductor()
        logger.info("Created PersonaConductor singleton")
    return _conductor_instance

def reset_conductor():
    """Reset the conductor state"""
    global _conductor_instance
    if _conductor_instance:
        _conductor_instance.reset_conversation()

# Legacy compatibility functions
_agent_instance: Optional[ConversationalAgent] = None

def get_conversational_agent() -> ConversationalAgent:
    """Legacy: Get singleton ConversationalAgent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
        logger.info("Created ConversationalAgent singleton (legacy)")
    return _agent_instance

# Aliases for compatibility
EnhancedPersonaConductor = PersonaConductor
EnhancedConversationalAgent = ConversationalAgent