# app.py - Simplified Version with Clean UI

import os
import streamlit as st
import logging
import sys
import asyncio
import time
import atexit
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD DEPENDENCIES
# ============================================================================

@st.cache_resource
def load_dependencies():
    """Load required dependencies"""
    from config import APP_TITLE, WELCOME_MESSAGE
    from conversational_agent import get_persona_conductor
    from guard import persona_validator
    from conversation import get_conversation_manager
    
    return {
        "APP_TITLE": APP_TITLE,
        "WELCOME_MESSAGE": WELCOME_MESSAGE,
        "get_persona_conductor": get_persona_conductor,
        "get_persona_validator": lambda: persona_validator,
        "get_conversation_manager": get_conversation_manager,
    }

# ============================================================================
# CLEANUP REGISTRATION
# ============================================================================

def register_cleanup():
    """Register cleanup handlers for graceful shutdown"""
    async def cleanup_resources():
        """Async cleanup of resources"""
        try:
            from llm_client import cleanup
            await cleanup()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def sync_cleanup():
        """Synchronous wrapper for cleanup"""
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(cleanup_resources())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(cleanup_resources())
        except Exception as e:
            logger.error(f"Sync cleanup failed: {e}")
    
    atexit.register(sync_cleanup)
    logger.info("Cleanup handlers registered")

# Load dependencies and register cleanup
try:
    deps = load_dependencies()
    register_cleanup()
    logger.info("All dependencies loaded successfully")
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
conductor = deps["get_persona_conductor"]()
validator = deps["get_persona_validator"]()

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
# QUERY HANDLER
# ============================================================================

async def handle_query(query: str) -> Dict[str, Any]:
    """Handle user query"""
    start_time = time.time()
    
    try:
        # Orchestrate response
        logger.info(f"Processing query: {query[:50]}...")
        decision = await conductor.orchestrate_response(query)
        
        # Validate if needed
        if decision.requires_validation:
            logger.info("Validating response...")
            validation = await validator.validate_response(
                response=decision.final_response,
                strategy_used=decision.strategy_used.value,
                context=decision.context_used,
                query=query
            )
            
            is_approved = validation.result.value == "approved"
            final_response = validation.final_response
            validation_info = {
                "result": validation.result.value,
                "confidence": validation.confidence
            }
        else:
            is_approved = True
            final_response = decision.final_response
            validation_info = {"result": "no_validation_needed"}
        
        # Update conversation history
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response)
        
        total_time = time.time() - start_time
        latency_ms = int(total_time * 1000)
        
        return {
            "success": True,
            "response": final_response,
            "approved": is_approved,
            "strategy": decision.strategy_used.value,
            "debug_info": {
                **decision.debug_info,
                "validation": validation_info,
                "latency_ms": latency_ms,
            }
        }
        
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "success": False,
            "response": "I apologize, but I encountered an error. Please try again.",
            "approved": False,
            "strategy": "error",
            "debug_info": {"error": str(e), "latency_ms": latency_ms}
        }

# ============================================================================
# SIDEBAR - SIMPLIFIED
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # New Conversation button
    if st.button("🔄 New Conversation", type="primary", use_container_width=True):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()
    
    # Clear Cache button
    if st.button("🗑️ Clear Response Cache", use_container_width=True):
        if conductor.cache:
            conductor.cache.cache.clear()
            st.success("Cache cleared")
        else:
            st.info("Cache is not enabled")
    
    # Debug mode checkbox
    debug_mode = st.checkbox("🔧 Debug Mode", value=False)

# ============================================================================
# MAIN CHAT UI
# ============================================================================

st.markdown("### 💬 Ask me anything about Journvax")

# Display chat history
for turn in conversation_manager.get_turns():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Handle user input
if query := st.chat_input("Type your question..."):
    # Display user message
    st.chat_message("user").write(query)
    
    # Process assistant response
    with st.chat_message("assistant"):
        # Process query with spinner
        with st.spinner("Thinking..."):
            result = run_async(handle_query(query))
        
        # Handle result
        if result["success"]:
            # Show warning if response was modified
            if not result.get("approved", True):
                st.warning("⚠️ Response modified for safety")
            
            # Display response
            st.write(result["response"])
            
            # Show debug info if enabled
            if debug_mode:
                with st.expander("🔧 Debug Information"):
                    # Show performance timing
                    if "latency_ms" in result.get("debug_info", {}):
                        st.metric("Response Time", f"{result['debug_info']['latency_ms']}ms")
                    
                    # Show strategy used
                    st.write(f"**Strategy:** {result.get('strategy', 'unknown')}")
                    
                    # Show if context was used
                    if "used_context" in result.get("debug_info", {}):
                        st.write(f"**Used Context:** {result['debug_info']['used_context']}")
                    
                    # Show full debug info
                    st.json(result.get("debug_info", {}))
        else:
            # Handle error
            st.error(result["response"])
            
            if debug_mode and "debug_info" in result:
                with st.expander("🔧 Error Details"):
                    st.json(result["debug_info"])