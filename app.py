# app.py

import os
import streamlit as st
import logging
import sys
import asyncio
import importlib

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

# --- CORE APPLICATION LOGIC ---
async def handle_query_async(query: str) -> dict:
    """
    Handles the user query via RAG -> LLM -> Guard.
    On model errors, shows a friendly message and does NOT write to history.
    """
    try:
        agent_decision = conversational_agent.process_query(query)

        # End-of-session handling
        if agent_decision.mode == deps["ConversationMode"].SESSION_END:
            conversation_manager.start_new_conversation()
            return {
                "success": True,
                "response": "Your session has ended. A new one has begun.",
                "approved": True,
                "context_str": "",
                "debug_info": agent_decision.debug_info,
            }

        # Greeting already handled by agent
        if not agent_decision.requires_generation:
            final_response_text = deps["WELCOME_MESSAGE"]
            is_approved = True
            guard_reasoning = "Not required for direct agent response."
        else:
            # Build prompt with retrieved context + history
            prompt_for_llm = deps["format_conversational_prompt"](
                query=query,
                formatted_context=agent_decision.context_str,
                conversation_context=conversation_manager.get_formatted_history(),
            )

            # Generate
            draft_response = await deps["call_base_assistant"](prompt_for_llm)

            # If model returns an error sentinel, short-circuit gracefully
            if draft_response.startswith("Error:"):
                return {
                    "success": False,
                    "response": "I'm sorry — there was a temporary issue contacting the model. Please try again.",
                    "approved": False,
                    "context_str": agent_decision.context_str,
                    "debug_info": {**agent_decision.debug_info, "model_error": draft_response},
                }

            # Guard validation
            is_approved, final_response_text, guard_reasoning = deps["evaluate_response"](
                context=agent_decision.context_str,
                user_question=query,
                assistant_response=draft_response,
                conversation_history=conversation_manager.get_formatted_history(),
            )

        # Write to history only on approved replies
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response_text)

        return {
            "success": True,
            "response": final_response_text,
            "approved": is_approved,
            "context_str": agent_decision.context_str,
            "debug_info": {**agent_decision.debug_info, "guard_reasoning": guard_reasoning},
        }

    except Exception as e:
        logger.error(f"Error in handle_query_async: {e}", exc_info=True)
        return {
            "success": False,
            "response": "I'm sorry, there was a critical error processing your request.",
            "approved": False,
            "context_str": "",
            "debug_info": {"error": str(e)},
        }

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("🔧 Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    show_context = st.checkbox("Show Retrieved Context", value=False)

    st.subheader("💬 Conversation")
    if st.button("New Conversation", type="primary"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()

# --- MAIN UI ---
st.markdown("### 💬 Ask me anything about Lexapro")

# Display chat history (conversation manager should seed WELCOME_MESSAGE on new session)
for turn in conversation_manager.get_turns():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Handle user input
if query := st.chat_input("Type your question…"):
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            result = run_async(handle_query_async(query))

        if result["success"]:
            if not result["approved"]:
                st.error("⚠️ **Safety Filter Active**")
                st.warning(result["response"])
                if debug_mode:
                    st.expander("🛡️ Guard Details").write(result.get("debug_info", {}).get("guard_reasoning", "No reason provided."))
            else:
                st.write(result["response"])
        else:
            st.error(result["response"])

        if show_context and result.get("context_str"):
            with st.expander("📚 Retrieved Context"):
                st.text(result["context_str"])

        if debug_mode:
            with st.expander("🔧 Debug Information"):
                st.json(result.get("debug_info", {}))
