import os
os.environ.pop('HF_ENDPOINT', None) 
import streamlit as st
import logging
import time
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
# This pattern avoids errors if a module is slow or fails to load.
def get_dependencies():
    from config import APP_TITLE
    from prompt import format_conversational_prompt, ACKNOWLEDGE_GAP_PROMPT
    from llm_client import call_base_assistant, get_hf_client
    from guard import evaluate_response
    from conversation import get_conversation_manager
    from conversational_agent import get_conversational_agent, ConversationMode
    return {
        "APP_TITLE": APP_TITLE,
        "format_conversational_prompt": format_conversational_prompt,
        "ACKNOWLEDGE_GAP_PROMPT": ACKNOWLEDGE_GAP_PROMPT,
        "call_base_assistant": call_base_assistant,
        "get_hf_client": get_hf_client,
        "evaluate_response": evaluate_response,
        "get_conversation_manager": get_conversation_manager,
        "get_conversational_agent": get_conversational_agent,
        "ConversationMode": ConversationMode
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

# Get lazy loaded instances
conversation_manager = deps["get_conversation_manager"]()
hf_client = deps["get_hf_client"]()
conversational_agent = deps["get_conversational_agent"]()

# --- ASYNC HELPER ---
def run_async(coro):
    """Helper to run async functions in Streamlit's sync context."""
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # If there's already a running loop, create a new task
        import concurrent.futures
        import threading
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # No loop running, we can use asyncio.run directly
        return asyncio.run(coro)

# --- CORE APPLICATION LOGIC ---
async def handle_query_async(query: str) -> dict:
    """
    Handles the user query using the robust "RAG-then-Check" workflow.
    """
    try:
        # Step 1: Let the new conversational agent process the query.
        agent_decision = conversational_agent.process_query(query)
        
        # Handle session end immediately
        if agent_decision.mode == deps["ConversationMode"].SESSION_END:
            conversation_manager.start_new_conversation()
            return {
                "success": True, "response": "Your session has ended. A new one has begun.", 
                "approved": True, "context_str": "", "debug_info": agent_decision.debug_info
            }

        # Step 2: Check if the agent determined that LLM generation is needed.
        if agent_decision.requires_generation:
            logger.info("Agent requires LLM generation. Proceeding...")
            
            # Step 3: Generate response using available context
            prompt_for_llm = deps["format_conversational_prompt"](
                query=query,
                formatted_context=agent_decision.context_str,
                conversation_context=conversation_manager.get_formatted_history()
            )

            # Step 4: Call the LLM to generate the draft response.
            draft_response = await deps["call_base_assistant"](prompt_for_llm)
            
            # Step 5: Run the draft response through the simplified guard.
            is_approved, final_response_text, guard_reasoning = deps["evaluate_response"](
                context=agent_decision.context_str,
                user_question=query,
                assistant_response=draft_response,
                conversation_history=conversation_manager.get_formatted_history()
            )
        else:
            # The agent handled the query directly (e.g., it was a greeting).
            logger.info("Agent handled query directly. Skipping generation and guard.")
            final_response_text = "Hello! How can I help you with our documentation today?"
            is_approved = True 
            guard_reasoning = "Not required for direct agent response."

        # Step 6: Update conversation history if the response was approved.
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
            "response": "I'm sorry, there was a critical error processing your request.",
            "approved": False, "context_str": "", "debug_info": {"error": str(e)}
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

# --- MAIN UI AND PROCESSING LOOP ---
st.markdown("### 💬 Ask me anything about our knowledge base")

# Display chat history
for turn in conversation_manager.get_turns():
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Handle user input
if query := st.chat_input("What would you like to know?"):
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            # Run async function in sync context
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
