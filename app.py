# app.py - Complete Version with All Improvements

import os
import streamlit as st
import logging
import sys
import asyncio
import time
import atexit
from typing import Dict, Any, Optional
from datetime import datetime

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
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Track and display performance metrics"""
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0,
            "cache_hits": 0,
            "slow_requests": 0,
        }
        self.request_history = []
        self.max_history = 100
    
    def record_request(self, success: bool, latency_ms: int, cached: bool = False):
        """Record a request's performance"""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.metrics["total_latency"] += latency_ms
        
        if cached:
            self.metrics["cache_hits"] += 1
        
        if latency_ms > 5000:  # Slow request threshold
            self.metrics["slow_requests"] += 1
        
        # Keep history
        self.request_history.append({
            "timestamp": datetime.now(),
            "success": success,
            "latency_ms": latency_ms,
            "cached": cached
        })
        
        # Trim history
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        total = max(1, self.metrics["total_requests"])
        
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": (self.metrics["successful_requests"] / total) * 100,
            "avg_latency_ms": self.metrics["total_latency"] / total,
            "cache_hit_rate": (self.metrics["cache_hits"] / total) * 100,
            "slow_request_rate": (self.metrics["slow_requests"] / total) * 100,
        }

# Initialize performance monitor
if "performance_monitor" not in st.session_state:
    st.session_state.performance_monitor = PerformanceMonitor()

# ============================================================================
# LAZY LOADING OF DEPENDENCIES
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
            # Try to get the running loop
            try:
                loop = asyncio.get_running_loop()
                # Schedule cleanup as a task
                loop.create_task(cleanup_resources())
            except RuntimeError:
                # No running loop, create one
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
        # Try to get existing loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running, need to handle differently
        loop = None
    
    if loop and loop.is_running():
        # We're in an async context, use thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # We can run directly
        return asyncio.run(coro)

# ============================================================================
# ERROR HANDLING WRAPPER
# ============================================================================

async def safe_execute(coro, fallback_response: str = "I apologize, but I encountered an error. Please try again."):
    """Execute a coroutine with error handling"""
    try:
        return await coro
    except asyncio.CancelledError:
        logger.warning("Request was cancelled")
        return {
            "success": False,
            "response": "Request was cancelled.",
            "approved": False,
            "debug_info": {"error": "CancelledError"}
        }
    except Exception as e:
        logger.error(f"Unexpected error in safe_execute: {e}", exc_info=True)
        return {
            "success": False,
            "response": fallback_response,
            "approved": False,
            "debug_info": {"error": str(e)}
        }

# ============================================================================
# PERSONA CONDUCTOR HANDLER (New Architecture)
# ============================================================================

async def handle_query_with_conductor(query: str) -> Dict[str, Any]:
    """Handler for new Persona Conductor architecture with improved error handling"""
    conductor = deps["get_persona_conductor"]()
    validator = deps["get_persona_validator"]()
    
    start_time = time.time()
    cached = False
    
    try:
        # Step 1: Orchestrate response
        logger.info(f"Orchestrating response for: {query[:50]}...")
        conductor_decision = await conductor.orchestrate_response(query)
        
        # Check if response was cached
        if conductor_decision.strategy_used.value == "cached":
            cached = True
        
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
        latency_ms = int(total_time * 1000)
        
        # Record performance
        st.session_state.performance_monitor.record_request(
            success=True,
            latency_ms=latency_ms,
            cached=cached
        )
        
        return {
            "success": True,
            "response": final_response,
            "approved": is_approved,
            "strategy": conductor_decision.strategy_used.value,
            "debug_info": {
                **conductor_decision.debug_info,
                "validation": validation_info,
                "latency_ms": latency_ms,
                "architecture": "persona_conductor",
                "cached": cached
            }
        }
        
    except Exception as e:
        logger.error(f"Error in conductor: {e}", exc_info=True)
        
        # Record failure
        latency_ms = int((time.time() - start_time) * 1000)
        st.session_state.performance_monitor.record_request(
            success=False,
            latency_ms=latency_ms
        )
        
        return {
            "success": False,
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "approved": False,
            "strategy": "error",
            "debug_info": {"error": str(e), "latency_ms": latency_ms}
        }

# ============================================================================
# LEGACY HANDLER (Original Architecture)
# ============================================================================

async def handle_query_legacy(query: str) -> Dict[str, Any]:
    """Handler for legacy architecture with improved error handling"""
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
            # Record failure
            latency_ms = int((time.time() - start_time) * 1000)
            st.session_state.performance_monitor.record_request(
                success=False,
                latency_ms=latency_ms
            )
            
            return {
                "success": False,
                "response": "I'm sorry, there was an issue generating a response. Please try again.",
                "approved": False,
                "debug_info": {"error": draft_response, "latency_ms": latency_ms},
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
        latency_ms = int(total_time * 1000)
        
        # Record performance
        st.session_state.performance_monitor.record_request(
            success=True,
            latency_ms=latency_ms
        )
        
        return {
            "success": True,
            "response": final_response_text,
            "approved": is_approved,
            "debug_info": {
                **agent_decision.debug_info,
                "guard_reasoning": guard_reasoning,
                "latency_ms": latency_ms,
                "architecture": "legacy"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in legacy handler: {e}", exc_info=True)
        
        # Record failure
        latency_ms = int((time.time() - start_time) * 1000)
        st.session_state.performance_monitor.record_request(
            success=False,
            latency_ms=latency_ms
        )
        
        return {
            "success": False,
            "response": "I apologize, but I encountered an error. Please try again.",
            "approved": False,
            "debug_info": {"error": str(e), "latency_ms": latency_ms}
        }

# ============================================================================
# UNIFIED QUERY HANDLER
# ============================================================================

async def handle_query(query: str) -> Dict[str, Any]:
    """Route to appropriate handler based on architecture"""
    if ARCHITECTURE == "persona_conductor":
        return await safe_execute(handle_query_with_conductor(query))
    else:
        return await safe_execute(handle_query_legacy(query))

# ============================================================================
# SIDEBAR UI
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings & Monitoring")
    
    # Architecture Info
    if ARCHITECTURE == "persona_conductor":
        st.success("✨ Persona Orchestra Active")
        
        debug_mode = st.checkbox("Debug Mode", value=False)
        show_personas = st.checkbox("Show Persona Breakdown", value=False)
        show_validation = st.checkbox("Show Validation Details", value=False)
        show_performance = st.checkbox("Show Performance Metrics", value=True)
        
    else:
        st.warning("⚠️ Legacy Architecture Active")
        st.info("Persona system not yet deployed")
        
        debug_mode = st.checkbox("Debug Mode", value=False)
        show_context = st.checkbox("Show Retrieved Context", value=False)
        show_performance = st.checkbox("Show Performance Metrics", value=True)
        show_personas = False
        show_validation = False
    
    # Performance Metrics
    if show_performance:
        st.subheader("📊 Performance Metrics")
        stats = st.session_state.performance_monitor.get_stats()
        
        if stats["total_requests"] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
                st.metric("Cache Hits", f"{stats['cache_hit_rate']:.1f}%")
            with col2:
                st.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
                st.metric("Slow Requests", f"{stats['slow_request_rate']:.1f}%")
        else:
            st.info("No requests yet")
    
    # Conversation Management
    st.subheader("💬 Conversation")
    if st.button("🔄 New Conversation", type="primary"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()
    
    # System Info
    with st.expander("ℹ️ System Info"):
        if ARCHITECTURE == "persona_conductor":
            st.markdown("""
            **Architecture:** Dynamic Persona Synthesis ✨
            
            **Active Personas:**
            - 🤗 Empathetic Companion
            - 📚 Information Navigator
            - 🎭 Bridge Synthesizer
            
            **Features:**
            - Response caching
            - Parallel processing
            - Request batching
            - Retry logic
            """)
        else:
            st.markdown("""
            **Architecture:** Legacy (Generate → Guard)
            
            **Components:**
            - Single LLM generation
            - Post-generation validation
            - Heuristic + LLM guard
            """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        
        **Slow responses?**
        - Check if caching is enabled
        - Verify A10G endpoint is responsive
        - Consider reducing token limits
        
        **Errors occurring?**
        - Check HF_TOKEN is valid
        - Verify endpoint URL is correct
        - Look at debug logs for details
        
        **Session issues?**
        - Try starting a new conversation
        - Refresh the page if needed
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
    
    # Process assistant response
    with st.chat_message("assistant"):
        # Choose spinner text based on architecture
        if ARCHITECTURE == "persona_conductor":
            spinner_text = "🎭 Orchestrating response..."
        else:
            spinner_text = "🤖 Thinking..."
        
        # Process query with spinner
        with st.spinner(spinner_text):
            result = run_async(handle_query(query))
        
        # Handle result
        if result["success"]:
            # Show warning if response was modified
            if not result.get("approved", True):
                st.warning("⚠️ Response modified for safety")
            
            # Display response
            st.write(result["response"])
            
            # Show latency if available
            if "latency_ms" in result.get("debug_info", {}):
                latency = result["debug_info"]["latency_ms"]
                if latency > 5000:
                    st.caption(f"⏱️ Response time: {latency/1000:.1f}s (slower than usual)")
                elif result.get("debug_info", {}).get("cached"):
                    st.caption(f"⚡ Cached response ({latency}ms)")
            
            # Show personas breakdown (new architecture only)
            if show_personas and ARCHITECTURE == "persona_conductor":
                if "strategy" in result:
                    with st.expander("🎭 Response Strategy"):
                        strategy = result["strategy"]
                        st.write(f"**Strategy:** {strategy}")
                        
                        if strategy == "synthesized":
                            st.write("Used all three personas:")
                            st.write("- 🤗 Empathetic Companion")
                            st.write("- 📚 Information Navigator")
                            st.write("- 🎭 Bridge Synthesizer")
                        elif strategy == "pure_empathy":
                            st.write("- 🤗 Empathetic Companion only")
                        elif strategy == "pure_facts":
                            st.write("- 📚 Information Navigator only")
                        elif strategy == "conversational":
                            st.write("- 💬 Light conversational response")
                        elif strategy == "cached":
                            st.write("- ⚡ Retrieved from cache")
            
            # Show validation details
            if show_validation and "validation" in result.get("debug_info", {}):
                with st.expander("🛡️ Validation Details"):
                    val = result["debug_info"]["validation"]
                    st.write(f"**Result:** {val.get('result', 'unknown')}")
                    if "confidence" in val:
                        st.write(f"**Confidence:** {val['confidence']:.2f}")
                    if "reasoning" in val:
                        st.write(f"**Reasoning:** {val['reasoning']}")
            
            # Show debug info
            if debug_mode:
                with st.expander("🔧 Debug Information"):
                    # Show performance timing
                    if "timing" in result.get("debug_info", {}):
                        st.write("**Performance Breakdown:**")
                        timing = result["debug_info"]["timing"]
                        for key, value in timing.items():
                            st.write(f"- {key}: {value}ms")
                    
                    # Show full debug info
                    st.json(result.get("debug_info", {}))
        else:
            # Handle error
            st.error(result["response"])
            
            if debug_mode and "debug_info" in result:
                with st.expander("🔧 Error Details"):
                    st.json(result["debug_info"])

# Footer
st.divider()
if ARCHITECTURE == "persona_conductor":
    st.caption("🎭 Powered by Dynamic Persona Synthesis • A10G Optimized")
else:
    st.caption("🤖 Powered by Legacy Architecture • Consider upgrading to Persona system")