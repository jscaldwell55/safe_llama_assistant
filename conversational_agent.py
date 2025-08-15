# conversational_agent.py - Refactored as the "Conductor" with Dynamic Persona Synthesis

import logging
import re
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

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

@dataclass
class CompositionComponents:
    empathy_component: str = ""
    facts_component: str = ""
    synthesized_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConductorDecision:
    final_response: str = ""
    requires_validation: bool = True
    strategy_used: ResponseStrategy = ResponseStrategy.CONVERSATIONAL
    components: Optional[CompositionComponents] = None
    context_used: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# THE CONDUCTOR - ORCHESTRATOR OF PERSONAS
# ============================================================================

class PersonaConductor:
    """
    The Conductor orchestrates multiple LLM personas to create
    harmonious, safe, and empathetic responses.
    """
    
    def __init__(self):
        self._llm_client = None
        self._retriever = None
        self._conversation_manager = None
        
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
        Step 1: Intuit the user's intent using LLM-based classification
        """
        from prompts import format_intent_classification_prompt
        
        try:
            prompt = format_intent_classification_prompt(query)
            llm_client = self._get_llm_client()
            
            # Quick, low-latency call for intent classification
            response = await llm_client(prompt)
            
            # Parse JSON response
            intent_data = self._parse_intent_response(response)
            
            # Determine strategy based on intents
            strategy = self._determine_strategy(intent_data)
            intent_data.strategy = strategy
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}", exc_info=True)
            # Fallback to heuristic analysis
            return self._heuristic_intent_analysis(query)
    
    def _parse_intent_response(self, response: str) -> IntentAnalysis:
        """Parse the LLM's intent classification response"""
        try:
            # Try to extract JSON from response
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
                    information_topics=data.get("information_topics", [])
                )
        except Exception as e:
            logger.warning(f"Failed to parse intent JSON: {e}")
        
        # Fallback parsing
        response_lower = response.lower()
        if "emotional" in response_lower:
            return IntentAnalysis(primary_intent=UserIntent.EMOTIONAL, needs_empathy=True)
        elif "information" in response_lower:
            return IntentAnalysis(primary_intent=UserIntent.INFORMATIONAL, needs_facts=True)
        else:
            return IntentAnalysis(primary_intent=UserIntent.CONVERSATIONAL)
    
    def _heuristic_intent_analysis(self, query: str) -> IntentAnalysis:
        """Fallback heuristic-based intent analysis"""
        q_lower = query.lower()
        
        # Emotional indicators
        emotional_words = ["worried", "scared", "anxious", "concerned", "afraid", 
                         "struggling", "difficult", "hard", "tough", "overwhelming"]
        emotional_indicators = [w for w in emotional_words if w in q_lower]
        
        # Information indicators
        info_words = ["what", "how", "when", "side effect", "dosage", "interaction",
                     "symptom", "treatment", "medication", "tell me", "explain"]
        info_indicators = [w for w in info_words if w in q_lower]
        
        # Personal sharing indicators
        personal_words = ["i am", "i'm", "i feel", "my doctor", "diagnosed", "taking"]
        is_personal = any(w in q_lower for w in personal_words)
        
        # Determine primary intent
        if emotional_indicators and info_indicators:
            primary = UserIntent.MIXED
            needs_empathy = True
            needs_facts = True
        elif emotional_indicators or is_personal:
            primary = UserIntent.EMOTIONAL if emotional_indicators else UserIntent.PERSONAL_SHARING
            needs_empathy = True
            needs_facts = bool(info_indicators)
        elif info_indicators or "?" in query:
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
            information_topics=info_indicators
        )
    
    def _determine_strategy(self, intent: IntentAnalysis) -> ResponseStrategy:
        """Determine the response strategy based on intent analysis"""
        if intent.needs_empathy and intent.needs_facts:
            return ResponseStrategy.SYNTHESIZED
        elif intent.needs_empathy and not intent.needs_facts:
            return ResponseStrategy.PURE_EMPATHY
        elif intent.needs_facts and not intent.needs_empathy:
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
        Step 2: Compose the response using appropriate personas
        """
        components = CompositionComponents()
        
        if intent.strategy == ResponseStrategy.SYNTHESIZED:
            # Parallel composition for mixed intents
            components = await self._synthesized_composition(query, intent, context)
            
        elif intent.strategy == ResponseStrategy.PURE_EMPATHY:
            # Empathetic companion only
            components.empathy_component = await self._get_empathetic_response(query, intent)
            components.synthesized_response = components.empathy_component
            
        elif intent.strategy == ResponseStrategy.PURE_FACTS:
            # Information navigator only
            components.facts_component = await self._get_factual_response(query, context)
            components.synthesized_response = components.facts_component
            
        else:  # CONVERSATIONAL
            # Light conversational response
            components.synthesized_response = await self._get_conversational_response(query)
        
        components.metadata = {
            "strategy": intent.strategy.value,
            "personas_used": self._get_personas_used(intent.strategy)
        }
        
        return components
    
    async def _synthesized_composition(
        self, 
        query: str, 
        intent: IntentAnalysis,
        context: str
    ) -> CompositionComponents:
        """
        Orchestrate the three-step synthesis process for mixed intents
        """
        from prompts import (
            format_empathetic_prompt,
            format_navigator_prompt,
            format_synthesizer_prompt
        )
        
        llm_client = self._get_llm_client()
        
        # Step 1 & 2: Parallel calls to Empathetic Companion and Information Navigator
        empathy_task = asyncio.create_task(
            self._get_empathetic_response(query, intent)
        )
        facts_task = asyncio.create_task(
            self._get_factual_response(query, context)
        )
        
        empathy_component, facts_component = await asyncio.gather(
            empathy_task, facts_task
        )
        
        # Step 3: Bridge Synthesizer
        synthesis_prompt = format_synthesizer_prompt(
            empathy_component=empathy_component,
            facts_component=facts_component
        )
        
        synthesized_response = await llm_client(synthesis_prompt)
        
        return CompositionComponents(
            empathy_component=empathy_component,
            facts_component=facts_component,
            synthesized_response=synthesized_response,
            metadata={
                "synthesis_method": "three_step_orchestration",
                "components_generated": 3
            }
        )
    
    async def _get_empathetic_response(self, query: str, intent: IntentAnalysis) -> str:
        """Get response from Empathetic Companion persona"""
        from prompts import format_empathetic_prompt
        
        # Build emotional context from query and indicators
        emotional_context = f"User expression: {query}"
        if intent.emotional_indicators:
            emotional_context += f"\nDetected emotions: {', '.join(intent.emotional_indicators)}"
        
        prompt = format_empathetic_prompt(emotional_context)
        llm_client = self._get_llm_client()
        
        response = await llm_client(prompt)
        return response.strip()
    
    async def _get_factual_response(self, query: str, context: str) -> str:
        """Get response from Information Navigator persona"""
        from prompts import format_navigator_prompt
        
        if not context or not context.strip():
            return "No relevant information found in documentation."
        
        prompt = format_navigator_prompt(query, context)
        llm_client = self._get_llm_client()
        
        response = await llm_client(prompt)
        return response.strip()
    
    async def _get_conversational_response(self, query: str) -> str:
        """Get light conversational response"""
        prompt = f"""You are a helpful assistant. Provide a brief, friendly response.
        
User: {query}
Assistant:"""
        
        llm_client = self._get_llm_client()
        response = await llm_client(prompt)
        return response.strip()
    
    def _get_personas_used(self, strategy: ResponseStrategy) -> List[str]:
        """Return list of personas used for a given strategy"""
        if strategy == ResponseStrategy.SYNTHESIZED:
            return ["empathetic_companion", "information_navigator", "bridge_synthesizer"]
        elif strategy == ResponseStrategy.PURE_EMPATHY:
            return ["empathetic_companion"]
        elif strategy == ResponseStrategy.PURE_FACTS:
            return ["information_navigator"]
        else:
            return ["conversational"]
    
    async def orchestrate_response(self, query: str) -> ConductorDecision:
        """
        Main orchestration method - the complete flow
        """
        try:
            # Check for simple greetings first (optimization)
            if self._is_simple_greeting(query):
                return ConductorDecision(
                    final_response="Hello! How can I help you today?",
                    requires_validation=False,
                    strategy_used=ResponseStrategy.CONVERSATIONAL,
                    debug_info={"reason": "simple_greeting"}
                )
            
            # Step 1: Analyze intent
            intent = await self.analyze_intent(query)
            
            # Step 2: Retrieve context if needed
            context = ""
            if intent.needs_facts:
                retriever = self._get_retriever()
                context = retriever(query)
            
            # Step 3: Compose response using appropriate personas
            components = await self.compose_response(query, intent, context)
            
            # Step 4: Prepare decision
            decision = ConductorDecision(
                final_response=components.synthesized_response,
                requires_validation=intent.needs_facts,  # Only validate if facts were included
                strategy_used=intent.strategy,
                components=components,
                context_used=context,
                debug_info={
                    "intent_analysis": {
                        "primary": intent.primary_intent.value,
                        "secondary": [i.value for i in intent.secondary_intents],
                        "needs_empathy": intent.needs_empathy,
                        "needs_facts": intent.needs_facts,
                        "emotional_indicators": intent.emotional_indicators,
                        "information_topics": intent.information_topics
                    },
                    "composition": components.metadata,
                    "context_length": len(context)
                }
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            return ConductorDecision(
                final_response="I apologize, but I'm having trouble processing your request. Please try again.",
                requires_validation=False,
                strategy_used=ResponseStrategy.CONVERSATIONAL,
                debug_info={"error": str(e)}
            )
    
    def _is_simple_greeting(self, query: str) -> bool:
        """Check if query is a simple greeting"""
        if not query or len(query) > 20:
            return False
        
        q_lower = query.lower().strip()
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        
        # Exact match or very simple greeting
        if q_lower in greetings:
            return True
        
        # Check if it's just greeting words
        words = re.findall(r'[a-z]+', q_lower)
        if len(words) <= 3 and all(w in greetings for w in words):
            return True
        
        return False

# ============================================================================
# LEGACY ADAPTER (ConversationalAgent compatibility)
# ============================================================================

class ConversationMode(Enum):
    """Legacy enum for backward compatibility"""
    GENERAL = "general"
    SESSION_END = "session_end"

@dataclass
class AgentDecision:
    """Legacy dataclass for backward compatibility"""
    mode: ConversationMode = ConversationMode.GENERAL
    requires_generation: bool = True
    context_str: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)

class ConversationalAgent:
    """
    Legacy adapter that wraps the new PersonaConductor
    to maintain backward compatibility with existing code
    """
    
    def __init__(self):
        self.conductor = PersonaConductor()
        self.greeting_words = {"hi", "hello", "hey"}
        self.greeting_phrases = {"good morning", "good afternoon", "good evening"}
    
    def _is_greeting(self, query: str) -> bool:
        """Legacy greeting detection"""
        return self.conductor._is_simple_greeting(query)
    
    def process_query(self, query: str) -> AgentDecision:
        """
        Legacy synchronous interface that wraps the async conductor
        """
        import asyncio
        
        # Run async orchestration in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the conductor
        decision = loop.run_until_complete(
            self.conductor.orchestrate_response(query)
        )
        
        # Convert to legacy format
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
    """Get the singleton PersonaConductor instance"""
    global _conductor_instance
    if _conductor_instance is None:
        _conductor_instance = PersonaConductor()
    return _conductor_instance

def get_conversational_agent() -> ConversationalAgent:
    """Get the legacy ConversationalAgent for backward compatibility"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ConversationalAgent()
    return _agent_instance