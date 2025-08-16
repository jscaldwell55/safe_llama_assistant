# conversational_agent.py - Simplified Version with Bridge Synthesizer Only

import logging
import re
import json
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================
# SIMPLE CACHE
# ============================================================================

class SimpleCache:
    """Simple response cache"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get_key(self, query: str, context: str = "") -> str:
        """Generate cache key"""
        combined = f"{query.lower().strip()}:{context[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        if key in self.cache:
            logger.info(f"[CACHE] Hit for key: {key[:8]}...")
            return self.cache[key]
        return None
    
    def put(self, key: str, value: str):
        """Store response"""
        if len(self.cache) >= self.max_size:
            # Remove oldest
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value
        logger.info(f"[CACHE] Stored key: {key[:8]}...")

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ResponseStrategy(Enum):
    SYNTHESIZED = "synthesized"
    CONVERSATIONAL = "conversational"
    CACHED = "cached"
    ERROR = "error"

@dataclass
class ConductorDecision:
    final_response: str = ""
    requires_validation: bool = True
    strategy_used: ResponseStrategy = ResponseStrategy.SYNTHESIZED
    context_used: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: int = 0

# ============================================================================
# SIMPLIFIED PERSONA CONDUCTOR
# ============================================================================

class PersonaConductor:
    """
    Simplified Conductor using only Bridge Synthesizer
    """
    
    def __init__(self):
        self._retriever = None
        
        # Simple cache
        from config import ENABLE_RESPONSE_CACHE, MAX_CACHE_SIZE
        self.cache = SimpleCache(max_size=MAX_CACHE_SIZE) if ENABLE_RESPONSE_CACHE else None
        
        # Pre-compiled patterns for instant responses
        self.greeting_patterns = re.compile(
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!?.,]*$',
            re.IGNORECASE
        )
        self.thanks_patterns = re.compile(
            r'^(thank|thanks|thank you|thx|ty)[\s!?.,]*$',
            re.IGNORECASE
        )
        
        # Instant responses
        self.instant_responses = {
            "hello": "Hello! How can I help you with information about Lexapro today?",
            "hi": "Hi there! What would you like to know about Lexapro?",
            "hey": "Hey! How can I help you?",
            "thank you": "You're welcome! Is there anything else about Lexapro I can help with?",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "goodbye": "Goodbye! Take care!",
            "bye": "Bye! Have a great day!",
        }
    
    def _get_retriever(self):
        """Lazy load RAG retriever"""
        if self._retriever is None:
            from rag import retrieve_and_format_context
            self._retriever = retrieve_and_format_context
        return self._retriever
    
    def should_retrieve_context(self, query: str) -> bool:
        """Determine if we need to retrieve context"""
        query_lower = query.lower()
        
        # Skip retrieval for pure greetings/thanks
        if self.greeting_patterns.match(query) or self.thanks_patterns.match(query):
            return False
        
        # Skip for very short affirmations
        if query_lower in ["yes", "no", "ok", "okay", "sure"]:
            return False
        
        # Retrieve for everything else (questions, requests, etc.)
        return True
    
    async def generate_synthesized_response(self, query: str, context: str) -> str:
        """
        Generate response using Bridge Synthesizer approach
        """
        from prompts import BRIDGE_SYNTHESIZER_SIMPLE_PROMPT
        from config import BRIDGE_SYNTHESIZER_PARAMS
        from llm_client import call_huggingface_with_retry
        
        # Format the prompt
        if context and context.strip():
            prompt = BRIDGE_SYNTHESIZER_SIMPLE_PROMPT.replace("{query}", query)
            prompt = prompt.replace("{context}", context)
        else:
            # No context version
            prompt = f"""You are a helpful pharmaceutical assistant. Provide a natural, conversational response.

User Question: {query}

Response:"""
        
        try:
            response = await call_huggingface_with_retry(prompt, BRIDGE_SYNTHESIZER_PARAMS)
            
            if response.startswith("Error:"):
                return "I apologize, but I'm having trouble generating a response. Please try again."
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    async def orchestrate_response(self, query: str) -> ConductorDecision:
        """
        Main orchestration - simplified version
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cache_key = self.cache.get_key(query)
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    return ConductorDecision(
                        final_response=cached_response,
                        requires_validation=False,
                        strategy_used=ResponseStrategy.CACHED,
                        total_latency_ms=int((time.time() - start_time) * 1000),
                        debug_info={"cache_hit": True}
                    )
            
            # Check for instant responses
            query_lower = query.lower().strip()
            if query_lower in self.instant_responses:
                response = self.instant_responses[query_lower]
                
                if self.cache:
                    cache_key = self.cache.get_key(query)
                    self.cache.put(cache_key, response)
                
                return ConductorDecision(
                    final_response=response,
                    requires_validation=False,
                    strategy_used=ResponseStrategy.CONVERSATIONAL,
                    total_latency_ms=int((time.time() - start_time) * 1000),
                    debug_info={"instant_response": True}
                )
            
            # Step 1: Retrieve context if needed
            context = ""
            rag_time = 0
            if self.should_retrieve_context(query):
                rag_start = time.time()
                try:
                    retriever = self._get_retriever()
                    context = retriever(query)
                    logger.info(f"Retrieved context: {len(context)} chars")
                except Exception as e:
                    logger.error(f"RAG retrieval failed: {e}")
                    context = ""
                rag_time = int((time.time() - rag_start) * 1000)
            
            # Step 2: Generate synthesized response
            gen_start = time.time()
            response = await self.generate_synthesized_response(query, context)
            gen_time = int((time.time() - gen_start) * 1000)
            
            # Cache successful responses
            if self.cache and response and not response.startswith("Error"):
                cache_key = self.cache.get_key(query, context)
                self.cache.put(cache_key, response)
            
            # Build decision
            total_time = int((time.time() - start_time) * 1000)
            
            return ConductorDecision(
                final_response=response,
                requires_validation=bool(context),  # Only validate if we used context
                strategy_used=ResponseStrategy.SYNTHESIZED,
                context_used=context,
                total_latency_ms=total_time,
                debug_info={
                    "timing": {
                        "rag_ms": rag_time,
                        "generation_ms": gen_time,
                        "total_ms": total_time
                    },
                    "context_length": len(context),
                    "used_context": bool(context)
                }
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            
            return ConductorDecision(
                final_response="I apologize, but I'm having trouble processing your request. Please try again.",
                requires_validation=False,
                strategy_used=ResponseStrategy.ERROR,
                total_latency_ms=int((time.time() - start_time) * 1000),
                debug_info={"error": str(e)}
            )

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

class ConversationMode(Enum):
    GENERAL = "general"
    SESSION_END = "session_end"

@dataclass
class AgentDecision:
    mode: ConversationMode = ConversationMode.GENERAL
    requires_generation: bool = True
    context_str: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

class ConversationalAgent:
    """Legacy adapter for backward compatibility"""
    
    def __init__(self):
        self.conductor = PersonaConductor()
    
    def process_query(self, query: str) -> AgentDecision:
        """Synchronous wrapper for async conductor"""
        import asyncio
        
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async method
            if loop.is_running():
                # We're already in an async context
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
_agent_instance: Optional[ConversationalAgent] = None

def get_persona_conductor() -> PersonaConductor:
    """Get singleton PersonaConductor instance"""
    global _conductor_instance
    if _conductor_instance is None:
        _conductor_instance = PersonaConductor()
        logger.info("Created simplified PersonaConductor singleton")
    return _conductor_instance

def get_conversational_agent() -> ConversationalAgent:
    """Get singleton ConversationalAgent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
        logger.info("Created ConversationalAgent singleton")
    return _agent_instance

# ============================================================================
# SIMPLIFIED INTENT CLASSES (for compatibility)
# ============================================================================

class UserIntent(Enum):
    INFORMATIONAL = "informational"
    CONVERSATIONAL = "conversational"

@dataclass
class IntentAnalysis:
    primary_intent: UserIntent = UserIntent.INFORMATIONAL
    needs_facts: bool = True
    strategy: ResponseStrategy = ResponseStrategy.SYNTHESIZED
    confidence: float = 0.8

@dataclass
class CompositionComponents:
    synthesized_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)