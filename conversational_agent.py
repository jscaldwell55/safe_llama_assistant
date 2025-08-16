# conversational_agent.py - Enhanced Version with Early Query Validation

import logging
import re
import json
import asyncio
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED CACHE WITH DEDUPLICATION
# ============================================================================

class EnhancedCache:
    """Enhanced cache with response deduplication"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.fact_cache = {}  # Cache for individual facts to avoid repetition
    
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
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value
        logger.info(f"[CACHE] Stored key: {key[:8]}...")
    
    def store_facts(self, topic: str, facts: List[str]):
        """Store extracted facts to avoid repetition"""
        self.fact_cache[topic.lower()] = facts
    
    def get_facts(self, topic: str) -> Optional[List[str]]:
        """Get previously extracted facts"""
        return self.fact_cache.get(topic.lower())

# ============================================================================
# RESPONSE DEDUPLICATOR
# ============================================================================

class ResponseDeduplicator:
    """Ensures consistent, non-repetitive responses"""
    
    def __init__(self):
        self.mentioned_facts = set()
        self.response_history = []
    
    def deduplicate_facts(self, response: str, topic: str) -> str:
        """Remove already-mentioned facts from response"""
        # Extract facts from response
        sentences = response.split('. ')
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Create fact signature
            fact_sig = self._get_fact_signature(sentence)
            
            if fact_sig not in self.mentioned_facts:
                self.mentioned_facts.add(fact_sig)
                unique_sentences.append(sentence)
            else:
                logger.debug(f"Skipping duplicate fact: {sentence[:50]}...")
        
        if unique_sentences:
            result = '. '.join(unique_sentences)
            if result and result[-1] not in '.!?':
                result += '.'
            return result
        
        return "I've already provided that information. Is there something else about Journvax you'd like to know?"
    
    def _get_fact_signature(self, sentence: str) -> str:
        """Generate a signature for a fact to detect duplicates"""
        # Normalize the sentence
        normalized = sentence.lower().strip()
        # Remove common variations
        normalized = re.sub(r'\b(including|such as|like|for example)\b', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def reset(self):
        """Reset deduplication state"""
        self.mentioned_facts.clear()
        self.response_history.clear()

# ============================================================================
# ENHANCED PERSONA CONDUCTOR
# ============================================================================

class ResponseStrategy(Enum):
    SYNTHESIZED = "synthesized"
    CONVERSATIONAL = "conversational"
    CACHED = "cached"
    BLOCKED = "blocked"  # New: for dangerous queries
    ERROR = "error"

@dataclass
class ConductorDecision:
    final_response: str = ""
    requires_validation: bool = True
    strategy_used: ResponseStrategy = ResponseStrategy.SYNTHESIZED
    context_used: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: int = 0
    was_blocked: bool = False  # New: track if query was blocked

class EnhancedPersonaConductor:
    """
    Enhanced Conductor with:
    1. Early query validation
    2. Response deduplication
    3. Better fact consistency
    4. Improved transparency
    """
    
    def __init__(self):
        self._retriever = None
        self.deduplicator = ResponseDeduplicator()
        
        from config import ENABLE_RESPONSE_CACHE, MAX_CACHE_SIZE
        self.cache = EnhancedCache(max_size=MAX_CACHE_SIZE) if ENABLE_RESPONSE_CACHE else None
        
        # Guard for early detection
        self._guard = None
        
        # Quick response patterns
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
    
    def _get_guard(self):
        """Lazy load guard"""
        if self._guard is None:
            from guard import enhanced_guard
            self._guard = enhanced_guard
        return self._guard
    
    def _get_retriever(self):
        """Lazy load RAG retriever"""
        if self._retriever is None:
            from rag import retrieve_and_format_context
            self._retriever = retrieve_and_format_context
        return self._retriever
    
    def should_retrieve_context(self, query: str) -> bool:
        """Determine if we need to retrieve context"""
        query_lower = query.lower()
        
        if self.greeting_patterns.match(query) or self.thanks_patterns.match(query):
            return False
        
        if query_lower in ["yes", "no", "ok", "okay", "sure"]:
            return False
        
        return True
    
    async def generate_synthesized_response(self, query: str, context: str) -> str:
        """Generate response with better prompting"""
        from prompts import ENHANCED_BRIDGE_PROMPT
        from config import BRIDGE_SYNTHESIZER_PARAMS
        from llm_client import call_huggingface_with_retry
        
        # Build enhanced prompt
        if context and context.strip():
            prompt = ENHANCED_BRIDGE_PROMPT.format(
                query=query,
                context=context
            )
        else:
            prompt = f"""You are a helpful pharmaceutical assistant specializing in Journvax information.

User Question: {query}

Since no specific documentation is available for this query, provide a helpful response that:
1. Acknowledges the limitation
2. Offers to help with other Journvax-related questions
3. Suggests what types of information you CAN provide

Response:"""
        
        try:
            response = await call_huggingface_with_retry(prompt, BRIDGE_SYNTHESIZER_PARAMS)
            
            if response.startswith("Error:"):
                return "I apologize, but I'm having trouble generating a response. Please try again."
            
            # Deduplicate facts if about common topics
            if any(topic in query.lower() for topic in ['side effect', 'dosage', 'usage', 'interaction']):
                response = self.deduplicator.deduplicate_facts(response, query)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    async def orchestrate_response(self, query: str) -> ConductorDecision:
        """Enhanced orchestration with early query validation"""
        start_time = time.time()
        
        try:
            # STEP 1: Early query validation (NEW)
            guard = self._get_guard()
            query_validation = await guard.validate_query(query)
            
            if query_validation is not None:
                # Query was flagged as dangerous/inappropriate
                logger.warning(f"Query blocked: {query_validation.threat_type.value}")
                
                return ConductorDecision(
                    final_response=query_validation.final_response,
                    requires_validation=False,  # Already validated
                    strategy_used=ResponseStrategy.BLOCKED,
                    total_latency_ms=int((time.time() - start_time) * 1000),
                    was_blocked=True,
                    debug_info={
                        "blocked_reason": query_validation.reasoning,
                        "threat_type": query_validation.threat_type.value
                    }
                )
            
            # STEP 2: Check cache
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
            
            # STEP 3: Check instant responses
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
            
            # STEP 4: Retrieve context if needed
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
            
            # STEP 5: Generate response
            gen_start = time.time()
            response = await self.generate_synthesized_response(query, context)
            gen_time = int((time.time() - gen_start) * 1000)
            
            # STEP 6: Cache successful responses
            if self.cache and response and not response.startswith("Error"):
                cache_key = self.cache.get_key(query, context)
                self.cache.put(cache_key, response)
            
            total_time = int((time.time() - start_time) * 1000)
            
            return ConductorDecision(
                final_response=response,
                requires_validation=True,  # Always validate synthesized responses
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
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.deduplicator.reset()
        logger.info("Conversation state reset")



# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_conductor_instance: Optional[EnhancedPersonaConductor] = None

def get_persona_conductor() -> EnhancedPersonaConductor:
    """Get singleton PersonaConductor instance"""
    global _conductor_instance
    if _conductor_instance is None:
        _conductor_instance = EnhancedPersonaConductor()
        logger.info("Created enhanced PersonaConductor singleton")
    return _conductor_instance

def reset_conductor():
    """Reset the conductor state"""
    global _conductor_instance
    if _conductor_instance:
        _conductor_instance.reset_conversation()

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

PersonaConductor = EnhancedPersonaConductor  # Alias for compatibility

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
    """Legacy adapter"""
    def __init__(self):
        self.conductor = get_persona_conductor()
    
    def process_query(self, query: str) -> AgentDecision:
        import asyncio
        
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

_agent_instance: Optional[ConversationalAgent] = None

def get_conversational_agent() -> ConversationalAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
    return _agent_instance