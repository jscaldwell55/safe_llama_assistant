# app.py

import os
import streamlit as st
import logging
import sys
import asyncio
import importlib
import time

# Configure logging FIRST before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.info("=== Starting Streamlit app initialization ===")

def _safe_import_module(modname: str):
    """
    Robust import that survives Streamlit hot-reload partial module states.
    If a module failed previously, sys.modules[modname] can be a stale placeholder.
    """
    try:
        return importlib.import_module(modname)
    except KeyError:
        # Clear any broken placeholder and try again
        sys.modules.pop(modname, None)
        return importlib.import_module(modname)
    except Exception:
        # One more hard refresh attempt
        sys.modules.pop(modname, None)
        return importlib.import_module(modname)

# --- LAZY-LOADING IMPORTS ---
def get_dependencies():
    from config import APP_TITLE, WELCOME_MESSAGE
    # prompts module (renamed from prompt.py)
    prompts_mod = _safe_import_module("prompts")
    format_conversational_prompt = getattr(prompts_mod, "format_conversational_prompt")
    ACKNOWLEDGE_GAP_PROMPT = getattr(prompts_mod, "ACKNOWLEDGE_GAP_PROMPT")

    llm_mod = _safe_import_module("llm_client")
    call_base_assistant = getattr(llm_mod, "call_base_assistant")
    stream_base_assistant = getattr(llm_mod, "stream_base_assistant", None)  # New streaming function

    guard_mod = _safe_import_module("guard")
    evaluate_response = getattr(guard_mod, "evaluate_response")

    conv_mod = _safe_import_module("conversation")
    get_conversation_manager = getattr(conv_mod, "get_conversation_manager")

    agent_mod = _safe_import_module("conversational_agent")
    get_conversational_agent = getattr(agent_mod, "get_conversational_agent")
    ConversationMode = getattr(agent_mod, "ConversationMode")

    return {
        "APP_TITLE": APP_TITLE,
        "WELCOME_MESSAGE": WELCOME_MESSAGE,
        "format_conversational_prompt": format_conversational_prompt,
        "ACKNOWLEDGE_GAP_PROMPT": ACKNOWLEDGE_GAP_PROMPT,
        "call_base_assistant": call_base_assistant,
        "stream_base_assistant": stream_base_assistant,
        "evaluate_response": evaluate_response,
        "get_conversation_manager": get_conversation_manager,
        "get_conversational_agent": get_conversational_agent,
        "ConversationMode": ConversationMode,
    }

try:
    deps = get_dependencies()
    logger.info("All dependencies loaded successfully.")
except Exception as e:
    logger.error(f"A critical module failed to import: {e}", exc_info=True)
    st.error(f"Fatal Error: A required module could not be loaded. Please check the logs. Error: {e}")
    st.stop()

# --- PAGE AND RESOURCE CONFIGURATION ---
st.set_page_config(
    page_title=deps["APP_TITLE"],
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title(deps["APP_TITLE"])

# Get lazy loaded singletons
conversation_manager = deps["get_conversation_manager"]()
conversational_agent = deps["get_conversational_agent"]()

# --- HELPER FUNCTIONS ---
def _is_likely_conversational(query: str) -> bool:
    """Quick check if query is likely conversational (no medical content expected)"""
    q_lower = query.lower()
    conversational_indicators = [
        "thank", "hello", "hi", "how are you", "goodbye", "bye",
        "what can you do", "who are you", "help me understand",
        "good morning", "good afternoon", "good evening"
    ]
    return any(ind in q_lower for ind in conversational_indicators) and len(query) < 50

def _format_conversational_only_prompt(query: str) -> str:
    """Simpler prompt for conversational queries"""
    return f"""You are a helpful pharmaceutical assistant. Respond naturally and conversationally.

User: {query}
Assistant:"""

# --- ASYNC HELPER ---
def run_async(coro):
    """Run an async coroutine safely from Streamlit's sync context."""
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

# --- OPTIMIZED PARALLEL PROCESSING ---
async def handle_query_async_parallel(query: str) -> dict:
    """
    Handles user query with PARALLEL processing for improved latency.
    Fixed to avoid Streamlit thread context warnings.
    """
    start_time = time.time()
    
    try:
        # Process query synchronously to avoid thread context issues
        agent_decision = conversational_agent.process_query(query)
        
        # Get conversation history
        conversation_history = conversation_manager.get_formatted_history()
        
        logger.info(f"RAG retrieval took {time.time() - start_time:.2f}s")

        # End-of-session handling
        if agent_decision.mode == deps["ConversationMode"].SESSION_END:
            conversation_manager.start_new_conversation()
            return {
                "success": True,
                "response": "Your session has ended. A new one has begun.",
                "approved": True,
                "context_str": "",
                "debug_info": agent_decision.debug_info,
                "latency": time.time() - start_time,
            }

        # Greeting already handled by agent
        if not agent_decision.requires_generation:
            return {
                "success": True,
                "response": deps["WELCOME_MESSAGE"],
                "approved": True,
                "context_str": "",
                "debug_info": agent_decision.debug_info,
                "latency": time.time() - start_time,
            }

        # Check if likely conversational
        is_likely_conv = _is_likely_conversational(query)
        
        # Optimize prompt based on query type
        if is_likely_conv and not agent_decision.context_str:
            # Skip heavy prompt formatting for simple conversational queries
            prompt_for_llm = _format_conversational_only_prompt(query)
        else:
            # Full RAG-based prompt
            prompt_for_llm = deps["format_conversational_prompt"](
                query=query,
                formatted_context=agent_decision.context_str,
                conversation_context=conversation_history,
            )

        # Generate response
        gen_start = time.time()
        draft_response = await deps["call_base_assistant"](prompt_for_llm)
        logger.info(f"Generation took {time.time() - gen_start:.2f}s")

        # Handle model errors
        if draft_response.startswith("Error:"):
            return {
                "success": False,
                "response": "I'm sorry — there was a temporary issue contacting the model. Please try again.",
                "approved": False,
                "context_str": agent_decision.context_str,
                "debug_info": {**agent_decision.debug_info, "model_error": draft_response},
                "latency": time.time() - start_time,
            }

        # Guard validation (optimized with skipping logic)
        guard_start = time.time()
        try:
            is_approved, final_response_text, guard_reasoning = deps["evaluate_response"](
                context=agent_decision.context_str,
                user_question=query,
                assistant_response=draft_response,
                conversation_history=conversation_history,
            )
            logger.info(f"Guard evaluation took {time.time() - guard_start:.2f}s")
        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}")
            # If guard fails, default to safety
            is_approved = False
            final_response_text = "I apologize, but I cannot provide that information. Please consult with a healthcare professional."
            guard_reasoning = f"Guard error: {str(e)}"

        # Add to history if approved (sequential to avoid thread issues)
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response_text)

        total_latency = time.time() - start_time
        logger.info(f"Total query processing took {total_latency:.2f}s")

        return {
            "success": True,
            "response": final_response_text,
            "approved": is_approved,
            "context_str": agent_decision.context_str,
            "debug_info": {
                **agent_decision.debug_info, 
                "guard_reasoning": guard_reasoning,
                "latency_ms": int(total_latency * 1000),
                "is_conversational": is_likely_conv,
            },
        }

    except Exception as e:
        logger.error(f"Error in handle_query_async: {e}", exc_info=True)
        return {
            "success": False,
            "response": "I'm sorry, there was a critical error processing your request.",
            "approved": False,
            "context_str": "",
            "debug_info": {"error": str(e)},
            "latency": time.time() - start_time,
        }

# --- STREAMING VERSION (if supported) ---
async def handle_query_streaming(query: str, placeholder):
    """
    Handle query with response streaming for better perceived latency.
    Falls back to regular processing if streaming not available.
    """
    if not deps.get("stream_base_assistant"):
        # Fall back to regular processing
        result = await handle_query_async_parallel(query)
        return result
    
    start_time = time.time()
    
    try:
        # Start RAG retrieval
        agent_decision = await asyncio.to_thread(
            conversational_agent.process_query, query
        )
        
        # Build prompt
        prompt_for_llm = deps["format_conversational_prompt"](
            query=query,
            formatted_context=agent_decision.context_str,
            conversation_context=conversation_manager.get_formatted_history(),
        )
        
        # Stream response
        accumulated_response = ""
        async for chunk in deps["stream_base_assistant"](prompt_for_llm):
            accumulated_response += chunk
            placeholder.write(accumulated_response + "▌")
        
        # Guard validation on complete response
        is_approved, final_response_text, guard_reasoning = deps["evaluate_response"](
            context=agent_decision.context_str,
            user_question=query,
            assistant_response=accumulated_response,
            conversation_history=conversation_manager.get_formatted_history(),
        )
        
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response_text)
        
        return {
            "success": True,
            "response": final_response_text,
            "approved": is_approved,
            "context_str": agent_decision.context_str,
            "debug_info": {
                **agent_decision.debug_info,
                "guard_reasoning": guard_reasoning,
                "streamed": True,
                "latency_ms": int((time.time() - start_time) * 1000),
            },
        }
        
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # Fall back to regular processing
        return await handle_query_async_parallel(query)

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("🔧 Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    show_context = st.checkbox("Show Retrieved Context", value=False)
    enable_streaming = st.checkbox("Enable Streaming (Beta)", value=False)
    
    if debug_mode:
        st.subheader("⚡ Performance")
        st.info("Parallel processing enabled\nLLM guard skipping active")

    st.subheader("💬 Conversation")
    if st.button("New Conversation", type="primary"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()

# --- MAIN UI ---
st.markdown("### 💬 Ask me anything about Lexapro")

# Display chat history
for turn in conversation_manager.get_turns():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Handle user input
if query := st.chat_input("Type your question…"):
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        if enable_streaming and deps.get("stream_base_assistant"):
            # Streaming mode
            placeholder = st.empty()
            with st.spinner(""):
                result = run_async(handle_query_streaming(query, placeholder))
            placeholder.empty()  # Clear the placeholder
        else:
            # Regular mode with spinner
            with st.spinner("🤖 Thinking..."):
                result = run_async(handle_query_async_parallel(query))

        if result["success"]:
            if not result["approved"]:
                st.error("⚠️ **Safety Filter Active**")
                st.warning(result["response"])
                if debug_mode:
                    st.expander("🛡️ Guard Details").write(
                        result.get("debug_info", {}).get("guard_reasoning", "No reason provided.")
                    )
            else:
                st.write(result["response"])
        else:
            st.error(result["response"])

        if show_context and result.get("context_str"):
            with st.expander("📚 Retrieved Context"):
                st.text(result["context_str"])

        if debug_mode:
            with st.expander("🔧 Debug Information"):
                debug_info = result.get("debug_info", {})
                if "latency_ms" in debug_info:
                    st.metric("Response Time", f"{debug_info['latency_ms']}ms")
                st.json(debug_info)