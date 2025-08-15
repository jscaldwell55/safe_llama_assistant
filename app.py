# app.py - Streamlined with Persona Conductor Architecture

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
logger.info("=== Starting Streamlit app with Persona Conductor ===")

# ============================================================================
# LAZY LOADING OF DEPENDENCIES
# ============================================================================

@st.cache_resource
def load_dependencies():
    """Load all dependencies with caching"""
    from config import APP_TITLE, WELCOME_MESSAGE
    
    # New Persona Conductor system
    from conversational_agent import get_persona_conductor, PersonaConductor
    from guard import persona_validator, PersonaValidator
    
    # Legacy systems for backward compatibility
    from conversation import get_conversation_manager
    from conversational_agent import get_conversational_agent, ConversationMode
    
    return {
        "APP_TITLE": APP_TITLE,
        "WELCOME_MESSAGE": WELCOME_MESSAGE,
        "get_persona_conductor": get_persona_conductor,
        "get_persona_validator": lambda: persona_validator,
        "get_conversation_manager": get_conversation_manager,
        "get_conversational_agent": get_conversational_agent,
        "ConversationMode": ConversationMode,
    }

try:
    deps = load_dependencies()
    logger.info("All dependencies loaded successfully.")
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
conductor = deps["get_persona_conductor"]()
validator = deps["get_persona_validator"]()
conversation_manager = deps["get_conversation_manager"]()

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
        # Use thread pool for existing loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)

# ============================================================================
# MAIN QUERY HANDLER - SIMPLIFIED WITH CONDUCTOR
# ============================================================================

async def handle_query_with_conductor(query: str) -> Dict[str, Any]:
    """
    Simplified query handler using the Persona Conductor architecture
    """
    start_time = time.time()
    
    try:
        # Step 1: Orchestrate response through the Conductor
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
            
            # Use validation result
            is_approved = validation_decision.result.value == "approved"
            final_response = validation_decision.final_response
            validation_info = {
                "result": validation_decision.result.value,
                "reasoning": validation_decision.reasoning,
                "confidence": validation_decision.confidence
            }
        else:
            # No validation needed (pure conversational/empathetic)
            is_approved = True
            final_response = conductor_decision.final_response
            validation_info = {"result": "no_validation_needed"}
        
        # Step 3: Update conversation history if approved
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response)
        
        # Step 4: Prepare response
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
                "personas_used": conductor_decision.debug_info.get("composition", {}).get("personas_used", [])
            }
        }
        
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        return {
            "success": False,
            "response": "I apologize, but I encountered an error processing your request.",
            "approved": False,
            "strategy": "error",
            "debug_info": {"error": str(e), "latency_ms": int((time.time() - start_time) * 1000)}
        }

# ============================================================================
# STREAMING SUPPORT (OPTIONAL)
# ============================================================================

async def handle_query_streaming(query: str, placeholder):
    """
    Streaming version - can be enhanced to stream from personas
    """
    # For now, just use regular handling
    # Future: Implement streaming from Information Navigator
    result = await handle_query_with_conductor(query)
    return result

# ============================================================================
# SIDEBAR UI
# ============================================================================

with st.sidebar:
    st.header("🎭 Persona Orchestra Settings")
    
    # Debug mode
    debug_mode = st.checkbox("Debug Mode", value=False)
    
    # Show persona breakdown
    show_personas = st.checkbox("Show Persona Breakdown", value=False)
    
    # Show validation details
    show_validation = st.checkbox("Show Validation Details", value=False)
    
    # Enable streaming (future feature)
    enable_streaming = st.checkbox("Enable Streaming (Beta)", value=False, disabled=True)
    
    if debug_mode:
        st.subheader("⚡ Performance")
        st.info("Persona Conductor: Active\nDynamic Synthesis: Enabled")
    
    # Conversation controls
    st.subheader("💬 Conversation")
    if st.button("New Conversation", type="primary"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()
    
    # System info
    with st.expander("ℹ️ System Info"):
        st.markdown("""
        **Architecture:** Dynamic Persona Synthesis
        
        **Personas:**
        - 🤗 Empathetic Companion
        - 📚 Information Navigator
        - 🎭 Bridge Synthesizer
        
        **Strategy Selection:**
        - Intent-based routing
        - Parallel composition
        - Smart validation
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
    # Display user message
    st.chat_message("user").write(query)
    
    # Process with assistant
    with st.chat_message("assistant"):
        with st.spinner("🎭 Orchestrating response..."):
            result = run_async(handle_query_with_conductor(query))
        
        if result["success"]:
            # Display response
            if not result["approved"]:
                st.warning("⚠️ Response modified for safety")
            
            st.write(result["response"])
            
            # Show persona breakdown if enabled
            if show_personas and "personas_used" in result["debug_info"]:
                personas = result["debug_info"]["personas_used"]
                if personas:
                    with st.expander("🎭 Personas Used"):
                        for persona in personas:
                            if persona == "empathetic_companion":
                                st.write("🤗 **Empathetic Companion** - Provided emotional support")
                            elif persona == "information_navigator":
                                st.write("📚 **Information Navigator** - Retrieved factual information")
                            elif persona == "bridge_synthesizer":
                                st.write("🎭 **Bridge Synthesizer** - Combined components seamlessly")
            
            # Show validation details if enabled
            if show_validation and "validation" in result["debug_info"]:
                validation = result["debug_info"]["validation"]
                with st.expander("✅ Validation Details"):
                    st.json(validation)
            
            # Show debug info if enabled
            if debug_mode:
                with st.expander("🔧 Debug Information"):
                    # Clean up debug info for display
                    debug_data = result["debug_info"].copy()
                    
                    # Show strategy
                    st.metric("Strategy", result["strategy"])
                    
                    # Show latency
                    if "latency_ms" in debug_data:
                        st.metric("Response Time", f"{debug_data['latency_ms']}ms")
                    
                    # Show intent analysis
                    if "intent_analysis" in debug_data:
                        st.subheader("Intent Analysis")
                        st.json(debug_data["intent_analysis"])
                    
                    # Show full debug data
                    st.subheader("Full Debug Data")
                    st.json(debug_data)
        else:
            st.error(result["response"])
            if debug_mode:
                st.error(f"Error: {result['debug_info'].get('error', 'Unknown error')}")

# ============================================================================
# FOOTER
# ============================================================================

if debug_mode:
    st.markdown("---")
    st.caption("🎭 Powered by Dynamic Persona Synthesis Architecture")
    st.caption(f"Response Strategy: Intent → Compose → Validate")