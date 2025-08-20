# conversational_agent.py - Document-Grounded Conversational Agent Fixed Version
"""
Simplified agent that:
1. Retrieves relevant documentation
2. Generates responses grounded in documentation
3. Applies safety validation
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
    requires_validation: bool = True
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
    """Simple cache for responses to avoid repeated processing"""
    
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
        """Store response in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = value
        logger.info(f"Cached response for key: {key[:8]}...")

# ============================================================================
# MAIN CONDUCTOR CLASS
# ============================================================================

class PersonaConductor:
    """
    Main orchestrator for response generation.
    Ensures all responses are grounded in documentation.
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
            logger.warning("Could not load cache config, cache disabled")
        
        # Quick responses for common greetings
        self.greeting_patterns = re.compile(
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!?.,]*$',
            re.IGNORECASE
        )
        self.thanks_patterns = re.compile(
            r'^(thank|thanks|thank you|thx|ty)[\s!?.,]*$',
            re.IGNORECASE
        )
        
        self.instant_responses = {
            "hello": "Hello! How can I help you with information about Journvax today?",
            "hi": "Hi there! What would you like to know about Journvax?",
            "hey": "Hey! How can I help you?",
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
                from guard import enhanced_guard
                self._guard = enhanced_guard
            except ImportError:
                logger.error("Could not import guard")
        return self._guard
    
    def should_retrieve_context(self, query: str) -> bool:
        """Determine if we need to retrieve context for this query"""
        query_lower = query.lower()
        
        # Don't retrieve for greetings or thanks
        if self.greeting_patterns.match(query) or self.thanks_patterns.match(query):
            return False
        
        # Don't retrieve for single-word responses unless medical
        if query_lower in ["yes", "no", "ok", "okay", "sure"]:
            return False
        
        return True
    
    async def generate_synthesized_response(self, query: str, context: str) -> str:
        """
        Generate response that MUST be grounded in context.
        Uses strict prompting to prevent hallucination.
        """
        from llm_client import call_huggingface_with_retry
        from config import BRIDGE_SYNTHESIZER_PARAMS
        
        # If no meaningful context, return standard no-info response
        if not context or len(context.strip()) < 50:
            return "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"
        
        # Build strict grounding prompt
        prompt = f"""You are a pharmaceutical information specialist providing information about Journvax.

CRITICAL RULES:
1. ONLY provide information that is EXPLICITLY stated in the documentation below
2. Do NOT add information from general knowledge
3. Do NOT make inferences or assumptions
4. If the documentation doesn't contain the answer, say: "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"

COMPLIANCE REQUIREMENTS:
- Never provide dosing advice beyond what's in the documentation
- Never suggest dose changes or administration timing
- When discussing side effects, simply present the information from the documentation
- Never use phrases like "don't worry", "should be fine", "generally safe"
- Never imply safety from absence ("doesn't mention X, so...")

Documentation Available:
{context}

User Question: {query}

Response (using ONLY the documentation above):"""
        
        try:
            response = await call_huggingface_with_retry(prompt, BRIDGE_SYNTHESIZER_PARAMS)
            
            if response.startswith("Error:"):
                return "I apologize, but I'm having trouble generating a response. Please try again."
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    async def orchestrate_response(self, query: str) -> ConductorDecision:
        """
        Main orchestration logic with safety-first approach.
        Flow: Query validation → Cache check → Context retrieval → Generation → Response validation
        """
        start_time = time.time()
        
        try:
            # Step 1: Pre-screen query for safety violations
            guard = self._get_guard()
            if guard:
                query_validation = await guard.validate_query(query)
                
                if query_validation is not None:
                    # Query is unsafe - return blocked response
                    logger.warning(f"Query blocked: {query_validation.reasoning}")
                    
                    return ConductorDecision(
                        final_response=query_validation.final_response,
                        requires_validation=False,  # Already validated
                        strategy_used=ResponseStrategy.BLOCKED,
                        was_blocked=True,
                        total_latency_ms=int((time.time() - start_time) * 1000),
                        debug_info={
                            "blocked_reason": query_validation.reasoning,
                            "violation": query_validation.violation.value if hasattr(query_validation, 'violation') else "unknown",
                            "validation": {
                                "result": "blocked",
                                "violation": query_validation.violation.value if hasattr(query_validation, 'violation') else "unknown"
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
                        requires_validation=False,
                        strategy_used=ResponseStrategy.CACHED,
                        total_latency_ms=int((time.time() - start_time) * 1000),
                        debug_info={
                            "cache_hit": True,
                            "validation": {"result": "cached_approved"}
                        }
                    )
            
            # Step 3: Check for instant responses (greetings, etc.)
            query_lower = query.lower().strip()
            if query_lower in self.instant_responses:
                response = self.instant_responses[query_lower]
                
                if self.cache and cache_key:
                    self.cache.put(cache_key, response)
                
                return ConductorDecision(
                    final_response=response,
                    requires_validation=False,
                    strategy_used=ResponseStrategy.CONVERSATIONAL,
                    total_latency_ms=int((time.time() - start_time) * 1000),
                    debug_info={
                        "instant_response": True,
                        "validation": {"result": "instant_approved"}
                    }
                )
            
            # Step 4: Retrieve context if needed
            context = ""
            rag_time = 0
            if self.should_retrieve_context(query):
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
            
            # Step 6: Validate generated response
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
                    logger.warning(f"Response corrected: {validation_result.reasoning}")
                    response = validation_result.final_response
                    # Don't cache corrected responses
                elif self.cache and cache_key:
                    # Cache only validated, approved responses
                    self.cache.put(cache_key, response)
            
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
            
            # Always include validation info if validation was performed
            if validation_result:
                debug_info["validation"] = {
                    "result": validation_result.result.value,
                    "grounding_score": getattr(validation_result, 'grounding_score', 0.0),
                    "violation": validation_result.violation.value if hasattr(validation_result, 'violation') else None,
                    "was_corrected": validation_result.result.value != "approved",
                    "confidence": getattr(validation_result, 'confidence', 0.0),
                    "reasoning": validation_result.reasoning if validation_result.result.value != "approved" else None
                }
            else:
                debug_info["validation"] = {
                    "result": "no_guard",
                    "reason": "Guard not available"
                }
            
            return ConductorDecision(
                final_response=response,
                requires_validation=False,  # Already validated internally
                strategy_used=ResponseStrategy.SYNTHESIZED,
                context_used=context,
                total_latency_ms=total_time,
                grounding_score=validation_result.grounding_score if validation_result else 0.0,
                debug_info=debug_info
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            
            return ConductorDecision(
                final_response="I apologize, but I'm having trouble processing your request. Please try again.",
                requires_validation=False,
                strategy_used=ResponseStrategy.ERROR,
                total_latency_ms=int((time.time() - start_time) * 1000),
                debug_info={
                    "error": str(e),
                    "validation": {"result": "error"}
                }
            )
    
    def reset_conversation(self):
        """Reset conversation state (for new conversations)"""
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
            # Run async orchestration in sync context
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