# app.py - Simplified Streamlit UI with Extensive Logging (with streaming support)

import os
import streamlit as st
import logging
import sys
import asyncio
import time
import atexit
from typing import Dict, Any, AsyncGenerator

# ============================================================================
# CONFIGURE EXTENSIVE LOGGING
# ============================================================================

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)

# Set specific loggers to appropriate levels
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.INFO)
logging.getLogger("faiss").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============================================================================
# LOAD DEPENDENCIES
# ============================================================================

@st.cache_resource
def load_dependencies():
    """Load required dependencies"""
    logger.info("Loading dependencies...")
    
    try:
        from config import APP_TITLE, WELCOME_MESSAGE
        from conversational_agent import get_orchestrator, reset_orchestrator, ResponseDecision, ResponseStrategy
        from conversation import get_conversation_manager
        from rag import get_rag_system, get_index_stats
        
        deps = {
            "APP_TITLE": APP_TITLE,
            "WELCOME_MESSAGE": WELCOME_MESSAGE,
            "get_orchestrator": get_orchestrator,
            "reset_orchestrator": reset_orchestrator,
            "get_conversation_manager": get_conversation_manager,
            "get_rag_system": get_rag_system,
            "get_index_stats": get_index_stats,
            "ResponseDecision": ResponseDecision, # Pass dataclass for type checking
            "ResponseStrategy": ResponseStrategy, # Pass enum for type checking
        }
        
        logger.info("All dependencies loaded successfully")
        return deps
        
    except Exception as e:
        logger.error(f"Failed to load dependencies: {e}", exc_info=True)
        raise

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
    
    # Log system initialization
    logger.info("="*60)
    logger.info("PHARMA ENTERPRISE ASSISTANT INITIALIZED")
    logger.info(f"App Title: {deps['APP_TITLE']}")
    
    # Check API key
    from config import ANTHROPIC_API_KEY
    if ANTHROPIC_API_KEY:
        logger.info(f"Anthropic API Key configured (length: {len(ANTHROPIC_API_KEY)})")
    else:
        logger.error("ANTHROPIC_API_KEY not configured!")
    
    # Log RAG stats
    index_stats = deps["get_index_stats"]()
    logger.info(f"RAG Index Stats: {index_stats}")
    logger.info("="*60)
    
except Exception as e:
    logger.error(f"Fatal Error during initialization: {e}", exc_info=True)
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
orchestrator = deps["get_orchestrator"]()
ResponseDecision = deps["ResponseDecision"] # Get dataclass from deps
ResponseStrategy = deps["ResponseStrategy"] # Get enum from deps


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
# QUERY HANDLER (now returns a decision object that might contain a generator)
# ============================================================================

async def handle_query(query: str) -> Dict[str, Any]:
    """Handle user query with extensive logging and return generator for streaming"""
    start_time = time.time()
    request_id = f"REQ_{int(start_time)}_{hash(query) % 10000}"
    
    logger.info(f"[{request_id}] START Processing query: '{query}'")
    
    try:
        # Get conversation history for context
        conversation_history = conversation_manager.get_turns()
        
        # Orchestrate response - this will now return a ResponseDecision object.
        # If the strategy is GENERATED, decision.final_response will be an AsyncGenerator.
        # Otherwise, it will be a string (cached, fallback, error).
        logger.info(f"[{request_id}] Calling orchestrator...")
        decision = await orchestrator.orchestrate_response(query, conversation_history)
        
        # Log initial decision details (before full streaming completes if applicable)
        logger.info(f"[{request_id}] Response Decision (initial):")
        logger.info(f"  - Strategy: {decision.strategy_used.value}")
        logger.info(f"  - Context chars: {len(decision.context_used)}")
        logger.info(f"  - Grounding score: {decision.grounding_score:.3f}")
        logger.info(f"  - Validated: {decision.was_validated}")
        logger.info(f"  - Validation result: {decision.validation_result}")
        logger.info(f"  - Cache hit: {decision.cache_hit}")
        logger.info(f"  - Latency (up to LLM init): {decision.latency_ms}ms")
        
        total_time_initial = time.time() - start_time
        logger.info(f"[{request_id}] END Request initial processing completed in {total_time_initial:.2f}s")
        
        # Get orchestrator stats for logging
        stats = orchestrator.get_stats()
        logger.info(f"[STATS] Orchestrator: {stats}")
        
        return {
            "success": True,
            "decision": decision # Pass the entire decision object
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] ERROR: {e}", exc_info=True)
        
        total_time_error = time.time() - start_time
        logger.info(f"[{request_id}] END Request failed after {total_time_error:.2f}s")
        
        return {
            "success": False,
            "response": "I'm sorry, I don't have any information on that. Can I assist you with something else?",
            "strategy": "error",
            "grounding_score": 0.0,
            "latency_ms": int(total_time_error * 1000),
            "error": str(e)
        }

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # New Conversation button
    if st.button("🔄 New Conversation", type="primary", use_container_width=True):
        logger.info("[UI] New Conversation requested")
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        logger.info("[UI] New conversation started")
        st.rerun()
    
    # Clear Cache button
    if st.button("🗑️ Clear Response Cache", use_container_width=True):
        logger.info("[UI] Clear Cache requested")
        deps["reset_orchestrator"]()
        st.success("Cache cleared")
        logger.info("[UI] Cache cleared successfully")
    
    st.divider()
    
    # Display system stats
    st.subheader("📊 System Stats")
    
    # RAG stats
    rag_stats = deps["get_index_stats"]()
    if rag_stats["index_loaded"]:
        st.success("✅ RAG Index Loaded")
        st.metric("Documents", rag_stats["documents"])
        st.metric("Total Chunks", rag_stats["total_chunks"])
    else:
        st.error("❌ RAG Index Not Loaded")
    
    # Orchestrator stats
    orch_stats = orchestrator.get_stats()
    if orch_stats["total_requests"] > 0:
        st.metric("Total Requests", orch_stats["total_requests"])
        st.metric("Cache Hit Rate", f"{orch_stats.get('cache_hit_rate', 0):.1%}")
        st.metric("Fallback Rate", f"{orch_stats.get('fallback_rate', 0):.1%}")

# ============================================================================
# MAIN CHAT UI (updated for streaming)
# ============================================================================

st.markdown("### 💬 Ask me anything about Journvax")

# Display chat history
for turn in conversation_manager.get_display_turns():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Handle user input
if query := st.chat_input("Type your question..."):
    logger.info(f"[UI] User input received: '{query}'")
    
    # Display user message
    st.chat_message("user").write(query)
    
    # Process assistant response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.write("Thinking...") # Show thinking state
        
        result = run_async(handle_query(query))
        
        if result["success"]:
            decision: ResponseDecision = result["decision"]
            final_assistant_response_content = ""

            if decision.strategy_used == ResponseStrategy.GENERATED and isinstance(decision.final_response, AsyncGenerator):
                # Live generation, stream the response
                logger.info("[UI] Streaming LLM response...")
                full_streamed_response = st.write_stream(decision.final_response)
                final_assistant_response_content = full_streamed_response
                # After streaming, decision._full_generated_response_text will hold the *guarded* text
                # and _final_validation_result will hold the actual result.
                # Use these for history update if the orchestrator's post-stream processor has updated them.
                # NOTE: A slight race condition might exist if these are accessed immediately after st.write_stream
                # as the generator might still be finishing its post-processing.
                # For safety and consistency, we ensure the orchestrator populates these fields.
                
                # Check if post-stream validation results are available on the decision object
                # (This relies on the orchestrator's _post_stream_processor updating the decision object)
                if decision._full_generated_response_text:
                    logger.debug("[UI] Using post-stream validated text for history.")
                    final_assistant_response_content = decision._full_generated_response_text
                else:
                    logger.warning("[UI] Post-stream validated text not found on decision object. Using raw streamed text for history.")

            else:
                # Cached, fallback, or error responses (non-streaming)
                st.write(decision.final_response)
                final_assistant_response_content = decision.final_response
            
            # Remove thinking message now that content is shown (or streamed)
            status_placeholder.empty()

            # Update conversation history with the *full* final response
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_assistant_response_content)
            
            # Show performance metrics in tooltip
            metrics = []
            if decision.cache_hit:
                metrics.append("📌 Cached")
            else:
                # For streaming, the grounding score and validation result are from post-stream
                grounding_score_display = decision._final_grounding_score if hasattr(decision, '_final_grounding_score') and decision._final_grounding_score else decision.grounding_score
                validation_result_display = decision._final_validation_result if hasattr(decision, '_final_validation_result') and decision._final_validation_result else decision.validation_result
                
                if grounding_score_display > 0 and validation_result_display == "approved":
                    metrics.append(f"🎯 Grounding: {grounding_score_display:.2f}")
                elif validation_result_display == "rejected":
                     metrics.append(f"⚠️ Guard Rejected ({grounding_score_display:.2f})")
                elif validation_result_display == "no_context":
                     metrics.append(f"❌ No Context")
                elif validation_result_display == "pending":
                     metrics.append(f"⏳ Validating...") # Still validating or validation failed to update
            
            # Latency from decision object is up to LLM initiation, not full stream.
            # The full duration is logged in orchestrator.
            metrics.append(f"⚡ {decision.latency_ms}ms (initial)") 
            
            if metrics:
                st.caption(" | ".join(metrics))
            
            logger.info(f"[UI] Response delivered successfully (streamed or direct)")
        else:
            # Handle error (non-streaming)
            status_placeholder.empty() # Clear thinking message
            st.error(result["response"])
            # The orchestrator's error path will add fallback to conversation history
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", result["response"])
            logger.error(f"[UI] Response error: {result.get('error', 'Unknown')}")

# Footer with instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    - **Ask questions** about Journvax medication, dosage, side effects, interactions, etc.
    - **New Conversation** clears the chat history to start fresh
    - **Clear Cache** removes cached responses to force fresh generation
    - The assistant only provides information from uploaded documentation
    - If information isn't available, you'll receive a polite fallback message
    """)

logger.info("[UI] Page render complete")