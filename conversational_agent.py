# conversational_agent.py - Complete Final Version with All Fixes

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
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceTimer:
    """Context manager for timing operations"""
    def __init__(self, operation_name: str, log_slow_threshold_ms: int = 5000):
        self.operation_name = operation_name
        self.start_time = None
        self.log_slow_threshold_ms = log_slow_threshold_ms
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        
        if elapsed_ms > self.log_slow_threshold_ms:
            logger.warning(f"[PERF] SLOW {self.operation_name}: {elapsed_ms}ms")
        else:
            logger.info(f"[PERF] {self.operation_name}: {elapsed_ms}ms")
        
        self.elapsed_ms = elapsed_ms

# ============================================================================
# ENHANCED RESPONSE CACHE
# ============================================================================

class ResponseCache:
    """LRU cache for common responses with TTL support"""
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_order = []
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def get_cache_key(self, query: str, context: str = "") -> str:
        """Generate cache key from query and context"""
        # Normalize query for better cache hits
        normalized_query = query.lower().strip()
        normalized_query = re.sub(r'\s+', ' ', normalized_query)
        normalized_query = re.sub(r'[^\w\s]', '', normalized_query)
        
        combined = f"{normalized_query}:{context[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response with TTL check"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if datetime.now() - entry["timestamp"] > timedelta(seconds=self.ttl_seconds):
                # Expired
                del self.cache[key]
                self.access_order.remove(key)
                self.cache_stats["expirations"] += 1
                self.cache_stats["misses"] += 1
                logger.info(f"[CACHE] Expired key: {key[:8]}...")
                return None
            
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.cache_stats["hits"] += 1
            logger.info(f"[CACHE] Hit for key: {key[:8]}... (hits: {self.cache_stats['hits']})")
            return entry["value"]
        
        self.cache_stats["misses"] += 1
        return None
    
    def put(self, key: str, value: str):
        """Store response in cache with timestamp"""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            self.cache_stats["evictions"] += 1
            logger.info(f"[CACHE] Evicted LRU key: {lru_key[:8]}...")
        
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }
        self.access_order.append(key)
        logger.info(f"[CACHE] Stored key: {key[:8]}... (size: {len(self.cache)})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / max(1, total_requests)) * 100
        
        return {
            **self.cache_stats,
            "size": len(self.cache),
            "hit_rate": hit_rate
        }
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()
        logger.info("[CACHE] Cleared all entries")

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
    ERROR_RECOVERY = "error_recovery"

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
    Enhanced Conductor with better intent detection and error handling
    """
    
    def __init__(self):
        self._llm_client = None
        self._retriever = None
        self._conversation_manager = None
        
        # Initialize caches
        from config import ENABLE_RESPONSE_CACHE, CACHE_TTL_SECONDS, MAX_CACHE_SIZE
        
        self.response_cache = ResponseCache(
            max_size=MAX_CACHE_SIZE,
            ttl_seconds=CACHE_TTL_SECONDS
        ) if ENABLE_RESPONSE_CACHE else None
        
        self.intent_cache = {}  # Simple intent cache
        
        # Pre-compiled patterns for fast intent detection
        self.greeting_patterns = re.compile(
            r'^(hi|hello|hey|good morning|good afternoon|good evening|greetings)[\s!?.,]*$',
            re.IGNORECASE
        )
        self.thanks_patterns = re.compile(
            r'^(thank|thanks|thank you|thx|ty|appreciate)[\s!?.,]*$',
            re.IGNORECASE
        )
        self.goodbye_patterns = re.compile(
            r'^(bye|goodbye|see you|farewell|take care)[\s!?.,]*$',
            re.IGNORECASE
        )
        
        # Simple affirmation/negation patterns
        self.simple_response_patterns = re.compile(
            r'^(yes|no|ok|okay|sure|maybe|perhaps)[\s!?.,]*$',
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
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
    
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
        Optimized intent analysis with better detection patterns
        """
        # Check intent cache first
        query_lower = query.lower().strip()
        cache_key = hashlib.md5(query_lower.encode()).hexdigest()[:16]
        
        if cache_key in self.intent_cache:
            logger.info("[CACHE] Intent cache hit")
            return self.intent_cache[cache_key]
        
        with PerformanceTimer("intent_analysis", log_slow_threshold_ms=2000):
            # Fast path for common patterns
            if self.greeting_patterns.match(query):
                intent = IntentAnalysis(
                    primary_intent=UserIntent.CONVERSATIONAL,
                    strategy=ResponseStrategy.CONVERSATIONAL,
                    confidence=1.0
                )
                self.intent_cache[cache_key] = intent
                return intent
            
            if self.thanks_patterns.match(query):
                intent = IntentAnalysis(
                    primary_intent=UserIntent.CONVERSATIONAL,
                    strategy=ResponseStrategy.CONVERSATIONAL,
                    confidence=1.0
                )
                self.intent_cache[cache_key] = intent
                return intent
            
            if self.goodbye_patterns.match(query):
                intent = IntentAnalysis(
                    primary_intent=UserIntent.CONVERSATIONAL,
                    strategy=ResponseStrategy.CONVERSATIONAL,
                    confidence=1.0
                )
                self.intent_cache[cache_key] = intent
                return intent
            
            # For simple yes/no/ok responses, check context
            if self.simple_response_patterns.match(query):
                # These should be informational if they're likely following up on a question
                intent = IntentAnalysis(
                    primary_intent=UserIntent.INFORMATIONAL,
                    needs_facts=True,
                    strategy=ResponseStrategy.PURE_FACTS,
                    confidence=0.7
                )
                self.intent_cache[cache_key] = intent
                return intent
            
            # Use LLM for complex intent analysis
            from prompts import format_intent_classification_prompt
            from config import INTENT_CLASSIFIER_PARAMS
            
            try:
                prompt = format_intent_classification_prompt(query)
                
                # Use optimized parameters for intent classification
                from llm_client import call_huggingface
                response = await call_huggingface(prompt, INTENT_CLASSIFIER_PARAMS)
                
                # Check for errors
                if response.startswith("Error:"):
                    logger.warning(f"Intent classification failed: {response}")
                    return self._heuristic_intent_analysis(query)
                
                intent = self._parse_intent_response(response)
                intent.strategy = self._determine_strategy(intent)
                
                # Cache the result
                self.intent_cache[cache_key] = intent
                return intent
                
            except Exception as e:
                logger.error(f"Intent analysis failed, using heuristics: {e}")
                return self._heuristic_intent_analysis(query)
    
    def _parse_intent_response(self, response: str) -> IntentAnalysis:
        """Parse LLM intent response with error handling"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Parse intent with validation
                primary = UserIntent(data.get("primary_intent", "informational").lower())
                secondary = []
                for intent_str in data.get("secondary_intents", []):
                    try:
                        secondary.append(UserIntent(intent_str.lower()))
                    except ValueError:
                        continue
                
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
        
        # Fallback to heuristic analysis
        return self._heuristic_intent_analysis(response)
    
    def _heuristic_intent_analysis(self, query: str) -> IntentAnalysis:
        """Fast heuristic-based intent analysis with better patterns"""
        q_lower = query.lower().strip()
        
        # Initialize indicators
        emotional_indicators = []
        info_indicators = []
        
        # Check emotional content
        emotional_words = ['worried', 'scared', 'anxious', 'depressed', 'sad', 'afraid', 
                          'struggling', 'nervous', 'stressed', 'overwhelmed', 'concern',
                          'fear', 'panic', 'upset', 'frightened', 'terrified']
        for word in emotional_words:
            if word in q_lower:
                emotional_indicators.append(word)
        
        # Check informational content - IMPROVED PATTERNS
        info_patterns = [
            (r'\b(what|how|when|why|where|who|which|whose)\b', 'question'),
            (r'\b(side effect|adverse|reaction|effects)\b', 'side_effects'),
            (r'\b(dosage|dose|mg|milligram|how much|amount)\b', 'dosage'),
            (r'\b(interaction|interact|mixing|combine|together)\b', 'interactions'),
            (r'\b(tell me|explain|describe|information|info|about|detail)\b', 'information_request'),
            (r'\b(ingredient|contain|composition|made of|consists|active|inactive)\b', 'ingredients'),
            (r'\b(work|mechanism|function|operate|action)\b', 'mechanism'),
            (r'\b(use|usage|indication|prescribed for|treat|treatment|therapy)\b', 'usage'),
            (r'\b(warning|caution|danger|risk|safety)\b', 'warnings'),
            (r'\b(contraindication|avoid|should not|must not)\b', 'contraindications'),
        ]
        
        for pattern, topic in info_patterns:
            if re.search(pattern, q_lower):
                info_indicators.append(topic)
        
        # Special cases for very short queries
        word_count = len(q_lower.split())
        
        # Very short queries about specific topics should be informational
        if word_count <= 5:
            short_info_keywords = ['effect', 'effects', 'ingredient', 'ingredients', 'contain', 
                                  'active', 'dose', 'dosage', 'interaction', 'warning',
                                  'work', 'mechanism', 'use', 'usage', 'side', 'lexapro']
            for keyword in short_info_keywords:
                if keyword in q_lower:
                    info_indicators.append('direct_info_request')
                    break
        
        # Check for question marks
        has_question = '?' in query
        
        # Determine intent
        has_emotion = len(emotional_indicators) > 0
        has_info = len(info_indicators) > 0 or has_question
        
        # Bias toward informational for unclear cases
        if not has_emotion and (has_info or word_count <= 6):
            primary = UserIntent.INFORMATIONAL
            needs_empathy = False
            needs_facts = True
        elif has_emotion and has_info:
            primary = UserIntent.MIXED
            needs_empathy = True
            needs_facts = True
        elif has_emotion:
            primary = UserIntent.EMOTIONAL
            needs_empathy = True
            needs_facts = False
        elif has_info:
            primary = UserIntent.INFORMATIONAL
            needs_empathy = False
            needs_facts = True
        else:
            # Only mark as conversational if it's truly conversational
            conversational_words = ['hello', 'hi', 'hey', 'thanks', 'thank', 'bye', 
                                   'goodbye', 'good morning', 'good afternoon', 'good evening']
            is_conversational = any(word in q_lower for word in conversational_words)
            
            if is_conversational:
                primary = UserIntent.CONVERSATIONAL
                needs_empathy = False
                needs_facts = False
            else:
                # Default to informational for unclear cases
                primary = UserIntent.INFORMATIONAL
                needs_empathy = False
                needs_facts = True
        
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
        Optimized response composition with error handling
        """
        components = CompositionComponents()
        components.timing = {}
        
        with PerformanceTimer(f"compose_{intent.strategy.value}") as timer:
            try:
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
                    
            except Exception as e:
                logger.error(f"Error composing response: {e}")
                components.synthesized_response = "I apologize, but I'm having trouble generating a response. Please try again."
                intent.strategy = ResponseStrategy.ERROR_RECOVERY
        
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
        Parallel composition with timeout and error handling
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
            ENABLE_PARALLEL_PERSONAS,
            PARALLEL_TIMEOUT_SECONDS
        )
        
        components = CompositionComponents()
        
        if ENABLE_PARALLEL_PERSONAS:
            # TRUE PARALLEL execution
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
                    asyncio.gather(empathy_task, facts_task, return_exceptions=True),
                    timeout=PARALLEL_TIMEOUT_SECONDS
                )
                
                # Handle exceptions in results
                if isinstance(empathy_component, Exception):
                    logger.error(f"Empathy generation failed: {empathy_component}")
                    empathy_component = "I understand your concern."
                
                if isinstance(facts_component, Exception):
                    logger.error(f"Facts generation failed: {facts_component}")
                    facts_component = "Please consult the documentation for more information."
                
                components.timing["parallel_ms"] = int((time.time() - start_parallel) * 1000)
                
            except asyncio.TimeoutError:
                logger.error(f"Parallel composition timed out after {PARALLEL_TIMEOUT_SECONDS}s")
                empathy_component = "I understand your concern."
                facts_component = "Please consult the documentation for more information."
                
                # Cancel remaining tasks
                empathy_task.cancel()
                facts_task.cancel()
        else:
            # Sequential fallback
            empathy_component = await self._get_empathetic_response_optimized(query, intent)
            facts_component = await self._get_factual_response_optimized(query, context)
        
        # Synthesis step
        start_synthesis = time.time()
        
        try:
            synthesis_prompt = format_synthesizer_prompt(
                empathy_component=empathy_component,
                facts_component=facts_component
            )
            
            from llm_client import call_huggingface
            synthesized = await call_huggingface(synthesis_prompt, BRIDGE_SYNTHESIZER_PARAMS)
            
            # Check for error
            if synthesized.startswith("Error:"):
                # Fallback to simple concatenation
                synthesized = f"{empathy_component} {facts_component}".strip()
                
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback to simple concatenation
            synthesized = f"{empathy_component} {facts_component}".strip()
        
        components.timing["synthesis_ms"] = int((time.time() - start_synthesis) * 1000)
        
        components.empathy_component = empathy_component
        components.facts_component = facts_component
        components.synthesized_response = synthesized
        
        return components
    
    async def _get_empathetic_response_optimized(self, query: str, intent: IntentAnalysis) -> str:
        """Optimized empathetic response generation with error handling"""
        try:
            from prompts import format_empathetic_prompt
            from config import EMPATHETIC_COMPANION_PARAMS
            from llm_client import call_huggingface_with_retry
            
            emotional_context = f"User expression: {query}"
            if intent.emotional_indicators:
                emotional_context += f"\nDetected emotions: {', '.join(intent.emotional_indicators)}"
            
            prompt = format_empathetic_prompt(emotional_context)
            response = await call_huggingface_with_retry(prompt, EMPATHETIC_COMPANION_PARAMS)
            
            if response.startswith("Error:"):
                return "I understand this is important to you."
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Empathetic response failed: {e}")
            return "I understand this is important to you."
    
    async def _get_factual_response_optimized(self, query: str, context: str) -> str:
        """Optimized factual response generation with error handling"""
        try:
            from prompts import format_navigator_prompt
            from config import INFORMATION_NAVIGATOR_PARAMS
            from llm_client import call_huggingface_with_retry
            
            if not context or not context.strip():
                return "I don't have specific information about that in the documentation."
            
            prompt = format_navigator_prompt(query, context)
            response = await call_huggingface_with_retry(prompt, INFORMATION_NAVIGATOR_PARAMS)
            
            if response.startswith("Error:"):
                return "Information is available in the documentation."
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Factual response failed: {e}")
            return "Please refer to the documentation for this information."
    
    async def _get_empathetic_response(self, query: str, intent: IntentAnalysis) -> str:
        """Standard empathetic response"""
        return await self._get_empathetic_response_optimized(query, intent)
    
    async def _get_factual_response(self, query: str, context: str) -> str:
        """Standard factual response"""
        return await self._get_factual_response_optimized(query, context)
    
    async def _get_conversational_response(self, query: str) -> str:
        """Fast conversational response with fallback"""
        # Check for instant responses
        query_lower = query.lower().strip()
        if query_lower in self.instant_responses:
            return self.instant_responses[query_lower]
        
        try:
            # Generate simple response
            prompt = f"User: {query}\nAssistant: I can help you with information about Lexapro. "
            from config import MODEL_PARAMS
            params = MODEL_PARAMS.copy()
            params["max_new_tokens"] = 50  # Very short for conversational
            
            from llm_client import call_huggingface
            response = await call_huggingface(prompt, params)
            
            if response.startswith("Error:"):
                return "I'm here to help with information about Lexapro. What would you like to know?"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Conversational response failed: {e}")
            return "I'm here to help with information about Lexapro. What would you like to know?"
    
    async def orchestrate_response(self, query: str) -> ConductorDecision:
        """
        Main orchestration with comprehensive error handling and monitoring
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Check response cache first
            if self.response_cache:
                cache_key = self.response_cache.get_cache_key(query)
                cached_response = self.response_cache.get(cache_key)
                if cached_response:
                    total_time = int((time.time() - start_time) * 1000)
                    return ConductorDecision(
                        final_response=cached_response,
                        requires_validation=False,
                        strategy_used=ResponseStrategy.CACHED,
                        total_latency_ms=total_time,
                        debug_info={
                            "cache_hit": True,
                            "cache_stats": self.response_cache.get_stats()
                        }
                    )
            
            # Check for instant responses
            query_lower = query.lower().strip()
            if query_lower in self.instant_responses:
                response = self.instant_responses[query_lower]
                
                # Cache it
                if self.response_cache:
                    cache_key = self.response_cache.get_cache_key(query)
                    self.response_cache.put(cache_key, response)
                
                total_time = int((time.time() - start_time) * 1000)
                return ConductorDecision(
                    final_response=response,
                    requires_validation=False,
                    strategy_used=ResponseStrategy.CONVERSATIONAL,
                    total_latency_ms=total_time,
                    debug_info={"instant_response": True}
                )
            
            # Step 1: Analyze intent
            intent_start = time.time()
            intent = await self.analyze_intent(query)
            intent_time = int((time.time() - intent_start) * 1000)
            
            # Step 2: Retrieve context if needed
            context = ""
            rag_time = 0
            if intent.needs_facts:
                rag_start = time.time()
                try:
                    retriever = self._get_retriever()
                    context = retriever(query)
                except Exception as e:
                    logger.error(f"RAG retrieval failed: {e}")
                    context = ""
                rag_time = int((time.time() - rag_start) * 1000)
            
            # Step 3: Compose response
            compose_start = time.time()
            components = await self.compose_response(query, intent, context)
            compose_time = int((time.time() - compose_start) * 1000)
            
            # Cache successful responses
            if self.response_cache and components.synthesized_response and not components.synthesized_response.startswith("Error"):
                cache_key = self.response_cache.get_cache_key(query, context)
                self.response_cache.put(cache_key, components.synthesized_response)
            
            # Build decision with detailed timing
            total_time = int((time.time() - start_time) * 1000)
            self.total_latency += total_time
            
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
                        "info_topics": intent.information_topics,
                    },
                    "timing": {
                        "intent_ms": intent_time,
                        "rag_ms": rag_time,
                        "compose_ms": compose_time,
                        "total_ms": total_time,
                        **components.timing
                    },
                    "context_length": len(context),
                    "performance_stats": {
                        "request_count": self.request_count,
                        "error_count": self.error_count,
                        "avg_latency": self.total_latency / max(1, self.request_count)
                    }
                }
            )
            
            # Log performance warning if slow
            from config import LOG_SLOW_REQUESTS_THRESHOLD_MS
            if total_time > LOG_SLOW_REQUESTS_THRESHOLD_MS:
                logger.warning(f"[PERF] Slow request: {total_time}ms for query: {query[:50]}")
            
            return decision
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            
            total_time = int((time.time() - start_time) * 1000)
            
            return ConductorDecision(
                final_response="I apologize, but I'm having trouble processing your request. Please try again.",
                requires_validation=False,
                strategy_used=ResponseStrategy.ERROR_RECOVERY,
                total_latency_ms=total_time,
                debug_info={
                    "error": str(e),
                    "error_count": self.error_count
                }
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
        logger.info("Created PersonaConductor singleton")
    return _conductor_instance

def get_conversational_agent() -> ConversationalAgent:
    """Get singleton ConversationalAgent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
        logger.info("Created ConversationalAgent singleton")
    return _agent_instance