# app.py
import os
import streamlit as st
import logging
import sys
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.info("=== Starting Streamlit app initialization ===")

def get_dependencies():
    from config import APP_TITLE, SYSTEM_MESSAGES
    from prompts import format_conversational_prompt, ACKNOWLEDGE_GAP_PROMPT
    from llm_client import call_base_assistant, get_hf_client, reset_hf_client
    from guard import evaluate_response
    from conversation import get_conversation_manager
    from conversational_agent import get_conversational_agent, ConversationMode
    from rag import build_index
    return {
        "APP_TITLE": APP_TITLE,
        "SYSTEM_MESSAGES": SYSTEM_MESSAGES,
        "format_conversational_prompt": format_conversational_prompt,
        "ACKNOWLEDGE_GAP_PROMPT": ACKNOWLEDGE_GAP_PROMPT,
        "call_base_assistant": call_base_assistant,
        "get_hf_client": get_hf_client,
        "reset_hf_client": reset_hf_client,
        "evaluate_response": evaluate_response,
        "get_conversation_manager": get_conversation_manager,
        "get_conversational_agent": get_conversational_agent,
        "ConversationMode": ConversationMode,
        "build_index": build_index,
    }

try:
    deps = get_dependencies()
    logger.info("All dependencies loaded successfully.")
except Exception as e:
    logger.error(f"A critical module failed to import: {e}", exc_info=True)
    st.error(f"Fatal Error: A required module could not be loaded. Please check the logs. Error: {e}")
    st.stop()

st.set_page_config(page_title=deps["APP_TITLE"], layout="centered", initial_sidebar_state="collapsed")
st.title(deps["APP_TITLE"])

conversation_manager = deps["get_conversation_manager"]()
hf_client = deps["get_hf_client"]()
conversational_agent = deps["get_conversational_agent"]()

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)

async def handle_query_async(query: str) -> dict:
    try:
        agent_decision = conversational_agent.process_query(query)

        if agent_decision.mode == deps["ConversationMode"].SESSION_END:
            conversation_manager.start_new_conversation()
            return {
                "success": True, "response": deps["SYSTEM_MESSAGES"]["session_end"],
                "approved": True, "context_str": "", "debug_info": agent_decision.debug_info
            }

        if agent_decision.requires_generation:
            if not agent_decision.context_str.strip():
                final_response_text = deps["SYSTEM_MESSAGES"]["no_context"]
                is_approved = True
                guard_reasoning = "No context available → gap acknowledgment returned without generation."
            else:
                prompt_for_llm = deps["format_conversational_prompt"](
                    query=query,
                    formatted_context=agent_decision.context_str,
                    conversation_context=conversation_manager.get_formatted_history()
                )
                draft_response = await deps["call_base_assistant"](prompt_for_llm)

                lower = (draft_response or "").lower()
                if lower.startswith("error:") or lower.startswith("configuration error:"):
                    return {
                        "success": False,
                        "response": deps["SYSTEM_MESSAGES"]["error"],
                        "approved": False,
                        "context_str": agent_decision.context_str,
                        "debug_info": {**agent_decision.debug_info, "model_error": draft_response}
                    }

                is_approved, final_response_text, guard_reasoning = deps["evaluate_response"](
                    context=agent_decision.context_str,
                    user_question=query,
                    assistant_response=draft_response,
                    conversation_history=conversation_manager.get_formatted_history()
                )
        else:
            final_response_text = "Hi! Fire away with a question about our docs."
            is_approved = True
            guard_reasoning = "Not required for direct agent response."

        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response_text)

        return {
            "success": True,
            "response": final_response_text,
            "approved": is_approved,
            "context_str": agent_decision.context_str,
            "debug_info": {**agent_decision.debug_info, "guard_reasoning": guard_reasoning}
        }

    except Exception as e:
        logger.error(f"Error in handle_query_async: {e}", exc_info=True)
        return {
            "success": False,
            "response": deps["SYSTEM_MESSAGES"]["error"],
            "approved": False, "context_str": "", "debug_info": {"error": str(e)}
        }

with st.sidebar:
    st.header("🔧 Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    show_context = st.checkbox("Show Retrieved Context", value=False)

    st.subheader("💬 Conversation")
    if st.button("New Conversation", type="primary"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()

    st.subheader("📚 Index")
    if st.button("Build / Refresh Index"):
        with st.spinner("Building FAISS index..."):
            deps["build_index"](force_rebuild=True)
        st.success("Index built successfully.")

    st.subheader("🧰 Maintenance")
    if st.button("Reload Model Client"):
        deps["reset_hf_client"]()
        st.success("Model client reloaded. New endpoint will be used on next message.")

# UPDATED header line below:
st.markdown("### 💬 Ask me anything about Lexapro")

# Display chat history
for turn in conversation_manager.get_turns():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Handle user input
if query := st.chat_input("What would you like to know?"):
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            result = run_async(handle_query_async(query))

        if result["success"]:
            if not result["approved"]:
                st.error("⚠️ **Safety Filter Active**")
                st.warning(result['response'])
                if debug_mode:
                    st.expander("🛡️ Guard Details").write(result.get('debug_info', {}).get('guard_reasoning', 'No reason provided.'))
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
