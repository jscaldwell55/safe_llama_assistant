import os
os.environ.pop('HF_ENDPOINT', None) 
import streamlit as st
import logging
import time
import sys

# Configure logging FIRST before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.info("=== Starting Streamlit app initialization ===")

# --- MODULE IMPORTS ---
# We will import everything needed for the new workflow.
start_time = time.time()
try:
    from config import APP_TITLE
    from prompt import format_conversational_prompt
    from llm_client import call_base_assistant, get_hf_client
    from guard import evaluate_response
    from conversation import get_conversation_manager
    from conversational_agent import get_conversational_agent, ConversationMode
    logger.info(f"=== All imports completed in {time.time() - start_time:.2f}s ===")
except Exception as e:
    logger.error(f"A critical module failed to import: {e}", exc_info=True)
    st.error(f"Fatal Error: A required module could not be loaded. Please check the logs. Error: {e}")
    st.stop()


# --- PAGE AND RESOURCE CONFIGURATION ---
logger.info("Configuring Streamlit page...")
st.set_page_config(
    page_title=APP_TITLE, 
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title(APP_TITLE)
logger.info("Streamlit page configured successfully")

# Get lazy loaded instances using the functions from the modules themselves
conversation_manager = get_conversation_manager()
hf_client = get_hf_client()
conversational_agent = get_conversational_agent()

# --- CORE APPLICATION LOGIC ---

def handle_conversational_query(query: str) -> dict:
    """
    Handles the user query using the robust "RAG-then-Check" workflow.
    This function is now the central orchestrator.
    """
    try:
        # Step 1: Let the new conversational agent process the query.
        # This single call now handles RAG retrieval and the Answerability Check.
        agent_response = conversational_agent.process_conversation(query)
        
        final_response_text = ""
        is_approved = False
        guard_reasoning = "N/A"
        
        # Handle session end immediately
        if agent_response.mode == ConversationMode.SESSION_END:
            conversation_manager.start_new_conversation()
            return {
                "success": True, "response": agent_response.text, "approved": True,
                "context_str": "", "session_ended": True, "debug_info": agent_response.debug_info
            }

        # Step 2: Check if the agent determined that LLM generation is needed.
        if agent_response.requires_generation:
            logger.info("Agent requires LLM generation. Proceeding...")
            
            # Step 3: Format the prompt with the CORRECT 3 arguments.
            prompt_for_llm = format_conversational_prompt(
                query=query,
                formatted_context=agent_response.context,
                conversation_context=conversation_manager.get_formatted_history()
            )
            
            # Step 4: Call the LLM to generate the draft response.
            draft_response = call_base_assistant(prompt_for_llm)
            
            # Step 5: Run the draft response through the simplified guard.
            # The new guard only needs the context and the draft response.
            is_approved, final_response_text, guard_reasoning = evaluate_response(
                context=agent_response.context,
                assistant_response=draft_response
            )
        else:
            # The agent handled the query directly (e.g., it was a greeting).
            # No generation or guard check is needed.
            logger.info("Agent handled query directly. Skipping generation and guard.")
            final_response_text = agent_response.text
            is_approved = True # Direct responses from the agent are pre-approved.
            guard_reasoning = "Not required for direct agent response."

        # Step 6: Update conversation history if the response was approved.
        if is_approved:
            conversation_manager.add_turn("user", query)
            conversation_manager.add_turn("assistant", final_response_text)
        
        return {
            "success": True,
            "response": final_response_text,
            "approved": is_approved,
            "context_str": agent_response.context,
            "debug_info": {
                **agent_response.debug_info,
                "guard_reasoning": guard_reasoning,
            }
        }
        
    except Exception as e:
        logger.error(f"Error in handle_conversational_query: {e}", exc_info=True)
        return {
            "success": False,
            "response": "I'm sorry, there was a critical error processing your request. Please check the logs.",
            "approved": False,
            "context_str": "",
            "debug_info": {"error": str(e)}
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
            result = handle_conversational_query(query)
        
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

        # Optional context and debug display
        if show_context and result.get("context_str"):
            with st.expander("📚 Retrieved Context"):
                st.text(result["context_str"])
        
        if debug_mode:
            with st.expander("🔧 Debug Information"):
                st.json(result.get("debug_info", {}))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🛡️ <b>Enterprise Safe Assistant</b></div>", 
    unsafe_allow_html=True
)
