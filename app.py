# app.py - Simplified Streamlit UI with Extensive Logging

import os
import streamlit as st
import logging
import sys
import asyncio
import time
import atexit
from typing import Dict, Any

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
        from conversational_agent import get_orchestrator, reset_orchestrator
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
    """Handle user query with extensive logging"""
    start_time = time.time()
    request_id = f"REQ_{int(start_time)}_{hash(query) % 10000}"
    
    logger.info(f"[{request_id}] START Processing query: '{query}'")
    
    try:
        # Get conversation history for context
        conversation_history = conversation_manager.get_turns()
        
        # Orchestrate response
        logger.info(f"[{request_id}] Calling orchestrator...")
        decision = await orchestrator.orchestrate_response(query, conversation_history)
        
        # Log decision details
        logger.info(f"[{request_id}] Response Decision:")
        logger.info(f"  - Strategy: {decision.strategy_used.value}")
        logger.info(f"  - Context chars: {len(decision.context_used)}")
        logger.info(f"  - Grounding score: {decision.grounding_score:.3f}")
        logger.info(f"  - Validated: {decision.was_validated}")
        logger.info(f"  - Validation result: {decision.validation_result}")
        logger.info(f"  - Cache hit: {decision.cache_hit}")
        logger.info(f"  - Latency: {decision.latency_ms}ms")
        
        # Update conversation history
        conversation_manager.add_turn("user", query)
        conversation_manager.add_turn("assistant", decision.final_response)
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] END Request completed in {total_time:.2f}s")
        
        # Get orchestrator stats for logging
        stats = orchestrator.get_stats()
        logger.info(f"[STATS] Orchestrator: {stats}")
        
        return {
            "success": True,
            "response": decision.final_response,
            "strategy": decision.strategy_used.value,
            "grounding_score": decision.grounding_score,
            "latency_ms": decision.latency_ms,
            "cache_hit": decision.cache_hit,
            "validation_result": decision.validation_result
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] ERROR: {e}", exc_info=True)
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] END Request failed after {total_time:.2f}s")
        
        return {
            "success": False,
            "response": "I'm sorry, I don't have any information on that. Can I assist you with something else?",
            "strategy": "error",
            "grounding_score": 0.0,
            "latency_ms": int(total_time * 1000),
            "error": str(e)
        }

# ============================================================================
# SIDEBAR - SIMPLIFIED
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
    
    # Simplified system status
    st.subheader("📊 System Status")
    
    # Essential status
    rag_stats = deps["get_index_stats"]()
    if rag_stats["index_loaded"]:
        st.success("✅ Ready")
        st.caption(f"{rag_stats['documents']} documents loaded")
    else:
        st.error("❌ Index Not Loaded")
    
    # Optional: Performance metric
    orch_stats = orchestrator.get_stats()
    if orch_stats["total_requests"] > 0:
        st.caption(f"Sessions today: {orch_stats['total_requests']}")

# ============================================================================
# MAIN CHAT UI
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
        # Process query with spinner
        with st.spinner("Thinking..."):
            result = run_async(handle_query(query))
        
        # Handle result
        if result["success"]:
            # Display response
            st.write(result["response"])
            
            # Show minimal performance metrics
            metrics = []
            if result.get("cache_hit"):
                metrics.append("📌 Cached")
            elif result.get("grounding_score", 0) > 0:
                metrics.append(f"Score: {result['grounding_score']:.2f}")
            
            if result.get("latency_ms"):
                metrics.append(f"⚡ {result['latency_ms']}ms")
            
            if metrics:
                st.caption(" | ".join(metrics))
            
            logger.info(f"[UI] Response delivered successfully")
        else:
            # Handle error
            st.error(result["response"])
            logger.error(f"[UI] Response error: {result.get('error', 'Unknown')}")

logger.info("[UI] Page render complete")