# app.py
import os
import streamlit as st
import logging
import sys
import asyncio

# Configure logging FIRST before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.info("=== Starting Streamlit app initialization ===")

# --- LAZY-LOADING IMPORTS ---
def get_dependencies():
    from config import APP_TITLE, WELCOME_MESSAGE, HF_INFERENCE_ENDPOINT
    # prompts module (renamed from prompt.py)
    from prompts import format_conversational_prompt, ACKNOWLEDGE_GAP_PROMPT
    # llm client exposes call_* helpers
    from llm_client import call_base_assistant, reset_hf_client
    from guard import evaluate_response
    from conversation import get_conversation_manager
    from conversational_agent import get_conversational_agent, ConversationMode
    return {
        "APP_TITLE": APP_TITLE,
        "WELCOME_MESSAGE": WELCOME_MESSAGE,
        "HF_INFERENCE_ENDPOINT": HF_INFERENCE_ENDPOINT,
        "format_conversational_prompt": format_conversational_prompt,
        "ACKNOWLEDGE_GAP_PROMPT": ACKNOWLEDGE_GAP_PROMPT,
        "call_base_assistant": call_base_assistant,
        "reset_hf_client": reset_hf_client,
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

# --- UTILS ---
def _mask_endpoint(ep: str) -> str:
    if not ep:
        return "(not set)"
    ep = ep.strip()
    if len(ep) <= 22:
        return ep
    # keep scheme + host prefix and last 8 chars
    # e.g., https://xyz...cloud → https://xyz…cloud/…c0ffee42
    try:
        scheme, rest = ep.split("://", 1)
        host = rest.split("/", 1)[0]
        tail = ep[-8:]
        return f"{scheme}://{host}…/{tail}"
    except Exception:
        return ep[:22] + "…"

# --- ASYNC HELPER ---
def run_async(coro):
    """Run an async coroutine safely from Streamlit's sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In case Streamlit already has a loop running, execute in a thread
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
            if isinstance(draft_response, str) and draft_response.startswith("Error:"):
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

    # Optional: endpoint rotation tool (kept off by default; uncomment if you want)
    # if debug_mode:
    #     st.subheader("🧠 Model")
    #     st.caption(f"Endpoint: {_mask_endpoint(deps['HF_INFERENCE_ENDPOINT'])}")
    #     if st.button("Reload Model Client"):
    #         try:
    #             deps["reset_hf_client"]()
    #             st.success("Model client reloaded.")
    #         except Exception as e:
    #             st.error(f"Failed to reload client: {e}")

# --- MAIN UI ---
st.markdown("### 💬 Ask me anything about Lexapro")

# Display chat history (conversation manager seeds WELCOME_MESSAGE on new session)
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
                st.json(result.get("debug_info", {}))
