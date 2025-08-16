# conversational_agent.py - A10G Optimized Persona Conductor

import logging
import re
import json
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceTimer:
    """Context manager for timing operations"""
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        logger.info(f"[PERF] {self.operation_name}: {elapsed_ms}ms")

# ============================================================================
# RESPONSE CACHE
# ============================================================================

class ResponseCache:
    """LRU cache for common responses"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get_cache_key(self, query: str, context: str = "") -> str:
        """Generate cache key from query and context"""
        combined = f"{query}:{context[:100]}"  # Use first 100 chars of context
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            logger.info(f"[CACHE] Hit for key: {key[:8]}...")
            return self.cache[key]
        return None
    
    def put(self, key: str, value: str):
        """Store response in cache"""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
        logger.info(f"[CACHE] Stored key: {key[:8]}...")

# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

class UserIntent(Enum):
    EMOTIONAL = "emotional"
    INFORMATIONAL = "informational"
    CONVERSATIONAL = "conversational"
    PERSONAL_SHARING = "personal_sharing"
    CLARIFICATION = "clarification"
    MIXED = "mixed"

class ResponseStrategy(Enum):
    PURE_EMPATHY = "pure_empathy"
    PURE_FACTS = "pure_facts"
    SYNTHESIZED = "synthesized"
    CONVERSATIONAL = "conversational"
    CACHED = "cached"
    SESSION_END = "session_end"

@dataclass
class IntentAnalysis:
    primary_intent: UserIntent
    secondary_intents: List[UserIntent] = field(default_factory=list)
    needs_empathy: bool = False
    needs_facts: bool = False
    emotional_indicators: List[str] = field(default_factory=list)
    information_topics: List[str] = field(default_factory=list)
    strategy: ResponseStrategy = ResponseStrategy.CONVERSATIONAL
    confidence: float = 0.0

@dataclass
class CompositionComponents:
    empathy_component: str = ""
    facts_component: str = ""
    synthesized_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, int] = field(default_factory=dict)

@dataclass
class ConductorDecision:
    final_response: str = ""
    requires_validation: bool = True
    strategy_used: ResponseStrategy = ResponseStrategy.CONVERSATIONAL
    components: Optional[CompositionComponents] = None
    context_used: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: int = 0

# ============================================================================
# OPTIMIZED PERSONA CONDUCTOR
# ============================================================================

class PersonaConductor:
    """
    A10G-optimized Conductor with caching, parallel processing, and performance monitoring
    """
    
    def __init__(self):
        self._llm_client = None
        self._retriever = None
        self._conversation_manager = None
        
        # Initialize caches
        self.response_cache = ResponseCache(max_size=100)
        self.intent_cache = {}  # Simple intent cache
        
        # Pre-compiled patterns for fast intent detection
        self.greeting_patterns = re.compile(
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!?]*$',
            re.IGNORECASE
        )
        self.thanks_patterns = re.compile(
            r'^(thank|thanks|thank you|thx|ty)[\s!?]*',
            re.IGNORECASE
        )
        
        # Common responses for instant return
        self.instant_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I assist you with?",
            "hey": "Hey! How can I help?",
            "thank you": "You're welcome! Is there anything else I can help with?",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "goodbye": "Goodbye! Take care!",
            "bye": "Bye! Have a great day!",
        }
    
    def _get_llm_client(self):
        """Lazy load LLM client"""
        if self._llm_client is None:
            from llm_client import call_base_assistant
            self._llm_client = call_base_assistant
        return self._llm_client
    
    def _get_retriever(self):
        """Lazy load RAG retriever"""
        if self._retriever is None:
            from rag import retrieve_and_format_context
            self._retriever = retrieve_and_format_context
        return self._retriever
    
    def _get_conversation_manager(self):
        """Lazy load conversation manager"""
        if self._conversation_manager is None:
            from conversation import get_conversation_manager
            self._conversation_manager = get_conversation_manager()
        return self._conversation_manager
    
    async def analyze_intent(self, query: str) -> IntentAnalysis:
        """
        Optimized intent analysis with caching and fast paths
        """
        # Check intent cache first
        query_lower = query.lower().strip()
        if query_lower in self.intent_cache:
            logger.info("[CACHE] Intent cache hit")
            return self.intent_cache[query_lower]
        
        with PerformanceTimer("intent_analysis"):
            # Fast path for common patterns
            if self.greeting_patterns.match(query):
                intent = IntentAnalysis(
                    primary_intent=UserIntent.CONVERSATIONAL,
                    strategy=ResponseStrategy.CONVERSATIONAL,
                    confidence=1.0
                )
                self.intent_cache[query_lower] = intent
                return intent
            
            if self.thanks_patterns.match(query):
                intent = IntentAnalysis(
                    primary_intent=UserIntent.CONVERSATIONAL,
                    strategy=ResponseStrategy.CONVERSATIONAL,
                    confidence=1.0
                )
                self.intent_cache[query_lower] = intent
                return intent
            
            # Use LLM for complex intent analysis
            from prompts import format_intent_classification_prompt
            from config import INTENT_CLASSIFIER_PARAMS
            
            try:
                prompt = format_intent_classification_prompt(query)
                llm_client = self._get_llm_client()
                
                # Use optimized parameters for intent classification
                from llm_client import call_huggingface
                response = await call_huggingface(prompt, INTENT_CLASSIFIER_PARAMS)
                
                intent = self._parse_intent_response(response)
                intent.strategy = self._determine_strategy(intent)
                
                # Cache the result
                self.intent_cache[query_lower] = intent
                return intent
                
            except Exception as e:
                logger.error(f"Intent analysis failed, using heuristics: {e}")
                return self._heuristic_intent_analysis(query)
    
    def _parse_intent_response(self, response: str) -> IntentAnalysis:
        """Parse LLM intent response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                primary = UserIntent(data.get("primary_intent", "conversational").lower())
                secondary = [UserIntent(i.lower()) for i in data.get("secondary_intents", [])]
                
                return IntentAnalysis(
                    primary_intent=primary,
                    secondary_intents=secondary,
                    needs_empathy=data.get("needs_empathy", False),
                    needs_facts=data.get("needs_facts", False),
                    emotional_indicators=data.get("emotional_indicators", []),
                    information_topics=data.get("information_topics", []),
                    confidence=data.get("confidence", 0.7)
                )
        except Exception as e:
            logger.warning(f"Failed to parse intent JSON: {e}")
        
        return self._heuristic_intent_analysis(response)
    
    def _heuristic_intent_analysis(self, query: str) -> IntentAnalysis:
        """Fast heuristic-based intent analysis"""
        q_lower = query.lower()
        
        # Pre-compiled patterns for speed
        emotional_indicators = []
        info_indicators = []
        
        # Check emotional content
        if re.search(r'\b(worried|scared|anxious|depressed|sad|afraid|struggling)\b', q_lower):
            emotional_indicators.append("emotional")
        
        # Check informational content
        if re.search(r'\b(what|how|when|why|side effect|dosage|interaction)\b', q_lower):
            info_indicators.append("informational")
        
        # Determine intent
        if emotional_indicators and info_indicators:
            primary = UserIntent.MIXED
            needs_empathy = True
            needs_facts = True
        elif emotional_indicators:
            primary = UserIntent.EMOTIONAL
            needs_empathy = True
            needs_facts = False
        elif info_indicators or '?' in query:
            primary = UserIntent.INFORMATIONAL
            needs_empathy = False
            needs_facts = True
        else:
            primary = UserIntent.CONVERSATIONAL
            needs_empathy = False
            needs_facts = False
        
        return IntentAnalysis(
            primary_intent=primary,
            needs_empathy=needs_empathy,
            needs_facts=needs_facts,
            emotional_indicators=emotional_indicators,
            information_topics=info_indicators,
            confidence=0.8
        )
    
    def _determine_strategy(self, intent: IntentAnalysis) -> ResponseStrategy:
        """Determine optimal response strategy"""
        if intent.needs_empathy and intent.needs_facts:
            return ResponseStrategy.SYNTHESIZED
        elif intent.needs_empathy:
            return ResponseStrategy.PURE_EMPATHY
        elif intent.needs_facts:
            return ResponseStrategy.PURE_FACTS
        else:
            return ResponseStrategy.CONVERSATIONAL
    
    async def compose_response(
        self, 
        query: str, 
        intent: IntentAnalysis,
        context: str = ""
    ) -> CompositionComponents:
        """
        Optimized response composition with parallel processing
        """
        components = CompositionComponents()
        components.timing = {}
        
        with PerformanceTimer(f"compose_{intent.strategy.value}"):
            if intent.strategy == ResponseStrategy.SYNTHESIZED:
                # Parallel composition for mixed intents
                components = await self._synthesized_composition_parallel(query, intent, context)
                
            elif intent.strategy == ResponseStrategy.PURE_EMPATHY:
                # Empathetic companion only
                start = time.time()
                components.empathy_component = await self._get_empathetic_response(query, intent)
                components.synthesized_response = components.empathy_component
                components.timing["empathy_ms"] = int((time.time() - start) * 1000)
                
            elif intent.strategy == ResponseStrategy.PURE_FACTS:
                # Information navigator only
                start = time.time()
                components.facts_component = await self._get_factual_response(query, context)
                components.synthesized_response = components.facts_component
                components.timing["facts_ms"] = int((time.time() - start) * 1000)
                
            else:  # CONVERSATIONAL
                # Light conversational response
                start = time.time()
                components.synthesized_response = await self._get_conversational_response(query)
                components.timing["conversational_ms"] = int((time.time() - start) * 1000)
        
        components.metadata = {
            "strategy": intent.strategy.value,
            "confidence": intent.confidence,
            "timing": components.timing
        }
        
        return components
    
    async def _synthesized_composition_parallel(
        self, 
        query: str, 
        intent: IntentAnalysis,
        context: str
    ) -> CompositionComponents:
        """
        Parallel composition for A10G optimization
        """
        from prompts import (
            format_empathetic_prompt,
            format_navigator_prompt,
            format_synthesizer_prompt
        )
        from config import (
            EMPATHETIC_COMPANION_PARAMS,
            INFORMATION_NAVIGATOR_PARAMS,
            BRIDGE_SYNTHESIZER_PARAMS,
            ENABLE_PARALLEL_PERSONAS
        )
        
        components = CompositionComponents()
        
        if ENABLE_PARALLEL_PERSONAS:
            # TRUE PARALLEL execution on A10G
            start_parallel = time.time()
            
            empathy_task = asyncio.create_task(
                self._get_empathetic_response_optimized(query, intent)
            )
            facts_task = asyncio.create_task(
                self._get_factual_response_optimized(query, context)
            )
            
            # Wait for both with timeout
            try:
                empathy_component, facts_component = await asyncio.wait_for(
                    asyncio.gather(empathy_task, facts_task),
                    timeout=10.0  # 10 second timeout
                )
                components.timing["parallel_ms"] = int((time.time() - start_parallel) * 1000)
            except asyncio.TimeoutError:
                logger.error("Parallel composition timed out")
                empathy_component = "I understand your concern."
                facts_component = "Please consult the documentation."
        else:
            # Sequential fallback
            empathy_component = await self._get_empathetic_response_optimized(query, intent)
            facts_component = await self._get_factual_response_optimized(query, context)
        
        # Synthesis step
        start_synthesis = time.time()
        synthesis_prompt = format_synthesizer_prompt(
            empathy_component=empathy_component,
            facts_component=facts_component
        )
        
        from llm_client import call_huggingface
        synthesized = await call_huggingface(synthesis_prompt, BRIDGE_SYNTHESIZER_PARAMS)
        components.timing["synthesis_ms"] = int((time.time() - start_synthesis) * 1000)
        
        components.empathy_component = empathy_component
        components.facts_component = facts_component
        components.synthesized_response = synthesized
        
        return components
    
    async def _get_empathetic_response_optimized(self, query: str, intent: IntentAnalysis) -> str:
        """Optimized empathetic response generation"""
        from prompts import format_empathetic_prompt
        from config import EMPATHETIC_COMPANION_PARAMS
        from llm_client import call_huggingface
        
        emotional_context = f"User expression: {query}"
        if intent.emotional_indicators:
            emotional_context += f"\nDetected emotions: {', '.join(intent.emotional_indicators)}"
        
        prompt = format_empathetic_prompt(emotional_context)
        response = await call_huggingface(prompt, EMPATHETIC_COMPANION_PARAMS)
        return response.strip()
    
    async def _get_factual_response_optimized(self, query: str, context: str) -> str:
        """Optimized factual response generation"""
        from prompts import format_navigator_prompt
        from config import INFORMATION_NAVIGATOR_PARAMS
        from llm_client import call_huggingface
        
        if not context or not context.strip():
            return "No relevant information found in documentation."
        
        prompt = format_navigator_prompt(query, context)
        response = await call_huggingface(prompt, INFORMATION_NAVIGATOR_PARAMS)
        return response.strip()
    
    async def _get_empathetic_response(self, query: str, intent: IntentAnalysis) -> str:
        """Standard empathetic response"""
        return await self._get_empathetic_response_optimized(query, intent)
    
    async def _get_factual_response(self, query: str, context: str) -> str:
        """Standard factual response"""
        return await self._get_factual_response_optimized(query, context)
    
    async def _get_conversational_response(self, query: str) -> str:
        """Fast conversational response"""
        # Check for instant responses
        query_lower = query.lower().strip()
        if query_lower in self.instant_responses:
            return self.instant_responses[query_lower]
        
        # Generate simple response
        prompt = f"User: {query}\nAssistant:"
        from config import MODEL_PARAMS
        params = MODEL_PARAMS.copy()
        params["max_new_tokens"] = 50  # Very short for conversational
        
        from llm_client import call_huggingface
        response = await call_huggingface(prompt, params)
        return response.strip()
    
    async def orchestrate_response(self, query: str) -> ConductorDecision:
        """
        Main orchestration with comprehensive performance tracking
        """
        start_time = time.time()
        
        try:
            # Check response cache first
            cache_key = self.response_cache.get_cache_key(query)
            cached_response = self.response_cache.get(cache_key)
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
                self.response_cache.put(cache_key, response)
                return ConductorDecision(
                    final_response=response,
                    requires_validation=False,
                    strategy_used=ResponseStrategy.CONVERSATIONAL,
                    total_latency_ms=int((time.time() - start_time) * 1000),
                    debug_info={"instant_response": True}
                )
            
            # Step 1: Analyze intent
            intent_start = time.time()
            intent = await self.analyze_intent(query)
            intent_time = int((time.time() - intent_start) * 1000)
            
            # Step 2: Retrieve context if needed (with caching)
            context = ""
            rag_time = 0
            if intent.needs_facts:
                rag_start = time.time()
                retriever = self._get_retriever()
                context = retriever(query)
                rag_time = int((time.time() - rag_start) * 1000)
            
            # Step 3: Compose response
            compose_start = time.time()
            components = await self.compose_response(query, intent, context)
            compose_time = int((time.time() - compose_start) * 1000)
            
            # Cache successful responses
            if components.synthesized_response:
                self.response_cache.put(cache_key, components.synthesized_response)
            
            # Build decision with detailed timing
            total_time = int((time.time() - start_time) * 1000)
            
            decision = ConductorDecision(
                final_response=components.synthesized_response,
                requires_validation=intent.needs_facts,
                strategy_used=intent.strategy,
                components=components,
                context_used=context,
                total_latency_ms=total_time,
                debug_info={
                    "intent_analysis": {
                        "primary": intent.primary_intent.value,
                        "confidence": intent.confidence,
                        "needs_empathy": intent.needs_empathy,
                        "needs_facts": intent.needs_facts,
                    },
                    "timing": {
                        "intent_ms": intent_time,
                        "rag_ms": rag_time,
                        "compose_ms": compose_time,
                        "total_ms": total_time,
                        **components.timing
                    },
                    "cache_key": cache_key[:8],
                    "context_length": len(context)
                }
            )
            
            # Log performance warning if slow
            from config import LOG_SLOW_REQUESTS_THRESHOLD_MS
            if total_time > LOG_SLOW_REQUESTS_THRESHOLD_MS:
                logger.warning(f"[PERF] Slow request: {total_time}ms for query: {query[:50]}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            return ConductorDecision(
                final_response="I apologize, but I'm having trouble processing your request. Please try again.",
                requires_validation=False,
                strategy_used=ResponseStrategy.CONVERSATIONAL,
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
        self.greeting_words = {"hi", "hello", "hey"}
        self.greeting_phrases = {"good morning", "good afternoon", "good evening"}
    
    def _is_greeting(self, query: str) -> bool:
        return self.conductor.greeting_patterns.match(query) is not None
    
    def process_query(self, query: str) -> AgentDecision:
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        decision = loop.run_until_complete(
            self.conductor.orchestrate_response(query)
        )
        
        return AgentDecision(
            mode=ConversationMode.GENERAL,
            requires_generation=True,
            context_str=decision.context_used,
            debug_info=decision.debug_info
        )

# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_conductor_instance: Optional[PersonaConductor] = None
_agent_instance: Optional[ConversationalAgent] = None

def get_persona_conductor() -> PersonaConductor:
    global _conductor_instance
    if _conductor_instance is None:
        _conductor_instance = PersonaConductor()
    return _conductor_instance

def get_conversational_agent() -> ConversationalAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
    return _agent_instance