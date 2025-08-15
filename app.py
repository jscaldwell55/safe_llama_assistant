# app.py - Safe Migration Version with Feature Detection

import os
import streamlit as st
import logging
import sys
import asyncio
import time
from typing import Dict, Any, Optional

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE DETECTION - Check which architecture is available
# ============================================================================

def detect_architecture():
    """Detect whether new Persona Conductor architecture is available"""
    try:
        from conversational_agent import get_persona_conductor, PersonaConductor
        logger.info("=== Starting Streamlit app with Persona Conductor ===")
        return "persona_conductor"
    except ImportError:
        logger.info("=== Starting Streamlit app with Legacy Architecture ===")
        return "legacy"

ARCHITECTURE = detect_architecture()

# ============================================================================
# LAZY LOADING OF DEPENDENCIES (with architecture detection)
# ============================================================================

@st.cache_resource
def load_dependencies():
    """Load dependencies based on available architecture"""
    from config import APP_TITLE, WELCOME_MESSAGE
    
    deps = {
        "APP_TITLE": APP_TITLE,
        "WELCOME_MESSAGE": WELCOME_MESSAGE,
        "architecture": ARCHITECTURE,
    }
    
    if ARCHITECTURE == "persona_conductor":
        # New architecture available
        from conversational_agent import get_persona_conductor, PersonaConductor
        from guard import persona_validator, PersonaValidator
        from conversation import get_conversation_manager
        from conversational_agent import get_conversational_agent, ConversationMode
        
        deps.update({
            "get_persona_conductor": get_persona_conductor,
            "get_persona_validator": lambda: persona_validator,
            "get_conversation_manager": get_conversation_manager,
            "get_conversational_agent": get_conversational_agent,
            "ConversationMode": ConversationMode,
        })
    else:
        # Legacy architecture
        from prompts import format_conversational_prompt, ACKNOWLEDGE_GAP_PROMPT
        from llm_client import call_base_assistant
        from guard import evaluate_response
        from conversation import get_conversation_manager
        from conversational_agent import get_conversational_agent, ConversationMode
        
        deps.update({
            "format_conversational_prompt": format_conversational_prompt,
            "ACKNOWLEDGE_GAP_PROMPT": ACKNOWLEDGE_GAP_PROMPT,
            "call_base_assistant": call_base_assistant,
            "evaluate_response": evaluate_response,
            "get_conversation_manager": get_conversation_manager,
            "get_conversational_agent": get_conversational_agent,
            "ConversationMode": ConversationMode,
        })
    
    return deps

try:
    deps = load_dependencies()
    logger.info(f"All dependencies loaded successfully. Architecture: {ARCHITECTURE}")
except Exception as e:
    logger.error(f"Failed to load dependencies: {e}", exc_info=True)
    st.error(f"Fatal Error: Could not load required modules. Error: {e}")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=deps["APP_TITLE"],
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title(deps["APP_TITLE"])

# Get singletons
conversation_manager = deps["get_conversation_manager"]()
conversational_agent = deps["get_conversational_agent"]()

# ============================================================================
# ASYNC EXECUTION HELPER
# ============================================================================

def run_async(coro):
    """Run an async coroutine from Streamlit's sync context"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)

# ============================================================================
# PERSONA CONDUCTOR HANDLER (New Architecture)
# ============================================================================

async def handle_query_with_conductor(query: str) -> Dict[str, Any]:
    """Handler for new Persona Conductor architecture"""
    conductor = deps["get_persona_conductor"]()
    validator = deps["get_persona_validator"]()
    
    start_time = time.time()
    
    try:
        # Step 1: Orchestrate response
        logger.info(f"Orchestrating response for: {query[:50]}...")
        conductor_decision = await conductor.orchestrate_response(query)
        
        # Step 2: Validate if needed
        if conductor_decision.requires_validation:
            logger.info("Validating response...")
            validation_decision = await validator.validate_response(
                response=conductor_decision.final_response,
                strategy_used=conductor_decision.strategy_used.value,
                components=conductor_decision.components.__dict__ if conductor_decision.components else None,
                context=conductor_decision.context_used,
                query=query
            )
            
            is_approved = validation_decision.result.value == "approved"
            final_response = validation_decision.final_response
            validation_info = {
                "result": validation_decision.result.value,
                "reasoning": validation_decision.reasoning,
                "confidence": validation_decision.confidence
            }
        else:
            is_approved = True
            final_response = conductor_decision.final_response
            validation_info = {"result": "no_validation_needed"}
        
        # Step 3: Update conversation history
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response)
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "response": final_response,
            "approved": is_approved,
            "strategy": conductor_decision.strategy_used.value,
            "debug_info": {
                **conductor_decision.debug_info,
                "validation": validation_info,
                "latency_ms": int(total_time * 1000),
                "architecture": "persona_conductor"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in conductor: {e}", exc_info=True)
        return {
            "success": False,
            "response": "I apologize, but I encountered an error.",
            "approved": False,
            "strategy": "error",
            "debug_info": {"error": str(e)}
        }

# ============================================================================
# LEGACY HANDLER (Original Architecture)
# ============================================================================

async def handle_query_legacy(query: str) -> Dict[str, Any]:
    """Handler for legacy architecture"""
    start_time = time.time()
    
    try:
        # Get decision from agent
        agent_decision = conversational_agent.process_query(query)
        
        # Get conversation history
        conversation_history = conversation_manager.get_formatted_history()
        
        # Handle different modes
        if agent_decision.mode == deps["ConversationMode"].SESSION_END:
            conversation_manager.start_new_conversation()
            return {
                "success": True,
                "response": "Your session has ended. A new one has begun.",
                "approved": True,
                "debug_info": agent_decision.debug_info,
            }
        
        if not agent_decision.requires_generation:
            return {
                "success": True,
                "response": deps["WELCOME_MESSAGE"],
                "approved": True,
                "debug_info": agent_decision.debug_info,
            }
        
        # Format prompt
        prompt_for_llm = deps["format_conversational_prompt"](
            query=query,
            formatted_context=agent_decision.context_str,
            conversation_context=conversation_history,
        )
        
        # Generate response
        draft_response = await deps["call_base_assistant"](prompt_for_llm)
        
        if draft_response.startswith("Error:"):
            return {
                "success": False,
                "response": "I'm sorry, there was an issue. Please try again.",
                "approved": False,
                "debug_info": {"error": draft_response},
            }
        
        # Guard validation
        is_approved, final_response_text, guard_reasoning = deps["evaluate_response"](
            context=agent_decision.context_str,
            user_question=query,
            assistant_response=draft_response,
            conversation_history=conversation_history,
        )
        
        # Update history if approved
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response_text)
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "response": final_response_text,
            "approved": is_approved,
            "debug_info": {
                **agent_decision.debug_info,
                "guard_reasoning": guard_reasoning,
                "latency_ms": int(total_time * 1000),
                "architecture": "legacy"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in legacy handler: {e}", exc_info=True)
        return {
            "success": False,
            "response": "I apologize, but I encountered an error.",
            "approved": False,
            "debug_info": {"error": str(e)}
        }

# ============================================================================
# UNIFIED QUERY HANDLER
# ============================================================================

async def handle_query(query: str) -> Dict[str, Any]:
    """Route to appropriate handler based on architecture"""
    if ARCHITECTURE == "persona_conductor":
        return await handle_query_with_conductor(query)
    else:
        return await handle_query_legacy(query)

# ============================================================================
# SIDEBAR UI
# ============================================================================

with st.sidebar:
    if ARCHITECTURE == "persona_conductor":
        st.header("🎭 Persona Orchestra Settings")
        st.success("✨ New Architecture Active!")
        
        debug_mode = st.checkbox("Debug Mode", value=False)
        show_personas = st.checkbox("Show Persona Breakdown", value=False)
        show_validation = st.checkbox("Show Validation Details", value=False)
        
        if debug_mode:
            st.info("Persona Conductor: Active\nDynamic Synthesis: Enabled")
    else:
        st.header("🔧 Settings")
        st.warning("Legacy Architecture Active")
        st.info("New Persona system not yet deployed")
        
        debug_mode = st.checkbox("Debug Mode", value=False)
        show_context = st.checkbox("Show Retrieved Context", value=False)
        show_personas = False
        show_validation = False
    
    st.subheader("💬 Conversation")
    if st.button("New Conversation", type="primary"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()
    
    # Architecture info
    with st.expander("ℹ️ System Info"):
        if ARCHITECTURE == "persona_conductor":
            st.markdown("""
            **Architecture:** Dynamic Persona Synthesis ✨
            
            **Personas:**
            - 🤗 Empathetic Companion
            - 📚 Information Navigator
            - 🎭 Bridge Synthesizer
            """)
        else:
            st.markdown("""
            **Architecture:** Legacy (Generate → Guard)
            
            **Components:**
            - Single LLM generation
            - Post-generation validation
            - Heuristic + LLM guard
            """)

# ============================================================================
# MAIN CHAT UI
# ============================================================================

st.markdown("### 💬 Ask me anything about Lexapro")

# Display chat history
for turn in conversation_manager.get_turns():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Handle user input
if query := st.chat_input("Type your question..."):
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        if ARCHITECTURE == "persona_conductor":
            spinner_text = "🎭 Orchestrating response..."
        else:
            spinner_text = "🤖 Thinking..."
        
        with st.spinner(spinner_text):
            result = run_async(handle_query(query))
        
        if result["success"]:
            if not result["approved"]:
                st.warning("⚠️ Response modified for safety")
            
            st.write(result["response"])
            
            # Show personas (only for new architecture)
            if show_personas and ARCHITECTURE == "persona_conductor":
                if "personas_used" in result.get("debug_info", {}):
                    with st.expander("🎭 Personas Used"):
                        personas = result["debug_info"]["personas_used"]
                        for persona in personas:
                            if persona == "empathetic_companion":
                                st.write("🤗 **Empathetic Companion**")
                            elif persona == "information_navigator":
                                st.write("📚 **Information Navigator**")
                            elif persona == "bridge_synthesizer":
                                st.write("🎭 **Bridge Synthesizer**")
            
            # Show context (legacy architecture)
            if ARCHITECTURE == "legacy" and show_context:
                if "context_retrieved_length" in result.get("debug_info", {}):
                    with st.expander("📚 Retrieved Context"):
                        st.write(f"Context length: {result['debug_info']['context_retrieved_length']}")
            
            # Debug info
            if debug_mode:
                with st.expander("🔧 Debug Information"):
                    if "latency_ms" in result.get("debug_info", {}):
                        st.metric("Response Time", f"{result['debug_info']['latency_ms']}ms")
                    st.json(result.get("debug_info", {}))
        else:
            st.error(result["response"])

# Footer
if ARCHITECTURE == "persona_conductor":
    st.caption("🎭 Powered by Dynamic Persona Synthesis")
else:
    st.caption("🤖 Powered by Legacy Architecture")