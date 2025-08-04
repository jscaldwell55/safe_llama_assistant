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

# Log each import separately to identify hanging
start_time = time.time()

try:
    logger.info("Importing config...")
    from config import APP_TITLE
    logger.info(f"Config imported successfully in {time.time() - start_time:.2f}s")
except Exception as e:
    logger.error(f"Failed to import config: {e}")
    raise

try:
    logger.info("Importing prompt module...")
    from prompt import format_conversational_prompt
    logger.info(f"Prompt module imported successfully")
except Exception as e:
    logger.error(f"Failed to import prompt: {e}")
    raise

try:
    logger.info("Importing llm_client functions...")
    from llm_client import call_base_assistant
    logger.info(f"LLM client functions imported successfully")
except Exception as e:
    logger.error(f"Failed to import llm_client: {e}")
    raise

try:
    logger.info("Importing guard module...")
    from guard import evaluate_response
    logger.info(f"Guard module imported successfully")
except Exception as e:
    logger.error(f"Failed to import guard: {e}")
    raise

try:
    logger.info("Importing context_formatter module...")
    from context_formatter import context_formatter
    logger.info(f"Context formatter imported successfully")
except Exception as e:
    logger.error(f"Failed to import context_formatter: {e}")
    raise

try:
    logger.info("Importing ConversationMode...")
    from conversational_agent import ConversationMode
    logger.info(f"ConversationMode imported successfully")
except Exception as e:
    logger.error(f"Failed to import conversational_agent: {e}")
    raise

logger.info(f"=== All imports completed in {time.time() - start_time:.2f}s ===")

# Page configuration
logger.info("Configuring Streamlit page...")
try:
    st.set_page_config(
        page_title=APP_TITLE, 
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.title(APP_TITLE)
    logger.info("Streamlit page configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Streamlit page: {e}")
    raise

# Lazy loading functions for heavy resources
@st.cache_resource(show_spinner=False)
def get_rag_system():
    """Lazy load the RAG system"""
    logger.info("Lazy loading RAG system...")
    start_time = time.time()
    from rag import rag_system
    logger.info(f"RAG system loaded in {time.time() - start_time:.2f}s")
    return rag_system

@st.cache_resource(show_spinner=False) 
def get_conversation_manager():
    """Lazy load the conversation manager"""
    logger.info("Lazy loading conversation manager...")
    start_time = time.time()
    from conversation import conversation_manager as cm
    logger.info(f"Conversation manager loaded in {time.time() - start_time:.2f}s")
    return cm

@st.cache_resource(show_spinner=False)
def get_hf_client():
    """Lazy load the HF client"""
    logger.info("Lazy loading HF client...")
    start_time = time.time()
    from llm_client import hf_client as hfc
    logger.info(f"HF client loaded in {time.time() - start_time:.2f}s")
    return hfc

@st.cache_resource(show_spinner=False)
def get_conversational_agent():
    """Lazy load the conversational agent"""
    logger.info("Lazy loading conversational agent...")
    start_time = time.time()
    from conversational_agent import conversational_agent as ca
    logger.info(f"Conversational agent loaded in {time.time() - start_time:.2f}s")
    return ca

# Get lazy loaded instances
conversation_manager = get_conversation_manager()
hf_client = get_hf_client()
conversational_agent = get_conversational_agent()

def handle_conversational_query(query: str):
    """
    Simplified query processing that trusts the model's conversational abilities.
    
    Args:
        query (str): The user's question
        
    Returns:
        dict: Response information including final answer, context, and debug info
    """
    try:
        # Step 1: Check for session management
        conv_response = conversational_agent.process_conversation(query)
        
        # Handle session end
        if conv_response.mode == ConversationMode.SESSION_END:
            conversation_manager.start_new_conversation()
            return {
                "success": True,
                "response": conv_response.text,
                "approved": True,
                "context": [],
                "intent": conv_response.mode.value,
                "topic": None,
                "session_ended": True,
                "debug_info": conv_response.debug_info
            }
        
        # Step 2: Get conversation context
        intent, topic = conversation_manager.classify_intent(query)
        conversation_context = conversation_manager.get_conversation_context()
        
        # Step 3: Get RAG content (already retrieved by conversational agent)
        enhanced_query = conv_response.debug_info.get("enhanced_query", query)
        rag_system = get_rag_system()
        context_chunks = rag_system.retrieve(enhanced_query)
        context_chunks = [result["text"] for result in context_chunks]
        
        # Step 4: Format context for the model
        conversation_entities = conversation_manager.conversation.active_entities if conversation_manager.conversation else []
        
        if context_chunks:
            formatted_context = context_formatter.format_enhanced_context(
                context_chunks, query, conversation_context, conversation_entities
            )
        else:
            formatted_context = ""
        
        # Step 5: Let the model generate naturally
        prompt = format_conversational_prompt(
            query, formatted_context, conversation_context, intent, topic
        )
        
        # Model generates response
        model_response = call_base_assistant(prompt)
        
        # Step 6: Guard evaluation
        context_text = "\n\n".join(context_chunks) if context_chunks else ""
        is_approved, final_response, guard_reasoning = evaluate_response(
            context_text, query, model_response, conversation_context
        )
        
        # Step 7: Update conversation history if approved
        if is_approved:
            conversation_manager.add_turn(query, final_response, context_chunks, topic)
        
        return {
            "success": True,
            "response": final_response,
            "approved": is_approved,
            "context": context_chunks,
            "intent": intent,
            "topic": topic,
            "debug_info": {
                "prompt": prompt if is_approved else "[Hidden due to rejection]",
                "model_response": model_response if is_approved else "[Hidden due to rejection]",
                "guard_reasoning": guard_reasoning,
                "enhanced_query": enhanced_query,
                "conversation_context": conversation_context,
                "has_rag_content": len(context_chunks) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "success": False,
            "response": "I'm sorry, there was an error processing your request. Please try again or start a new conversation.",
            "approved": False,
            "context": [],
            "debug_info": {"error": str(e)}
        }

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    show_context = st.checkbox("Show Retrieved Context", value=False)
    
    # Conversation controls
    st.subheader("💬 Conversation")
    if st.button("New Conversation", type="primary"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()
    
    # Show conversation stats with session limit info
    if conversation_manager.conversation:
        turn_count = len(conversation_manager.conversation.turns)
        turns_remaining = conversation_manager.get_turns_remaining()
        
        if turn_count > 0:
            if turns_remaining > 0:
                st.info(f"📊 Turns: {turn_count}/{conversation_manager.max_turns}")
                if turns_remaining <= 3:
                    st.warning(f"⚠️ Only {turns_remaining} turns remaining!")
            else:
                st.error("🔴 Session limit reached")
                
            if conversation_manager.conversation.current_topic:
                st.caption(f"💬 Topic: {conversation_manager.conversation.current_topic}")
                
            if conversation_manager.conversation.active_entities:
                entities_display = ", ".join(conversation_manager.conversation.active_entities[-3:])
                st.caption(f"🔍 Entities: {entities_display}")
        else:
            st.info("💫 Ready to start conversation")
    
    # Health check
    with st.expander("🏥 System Status"):
        if st.button("Check Health"):
            with st.spinner("Checking system health..."):
                # Check HF endpoint
                hf_healthy = hf_client.health_check()
                if hf_healthy:
                    st.success("✅ LLM endpoint healthy")
                else:
                    st.error("❌ LLM endpoint not responding")
                
                # Check RAG system
                rag_system = get_rag_system()
                if rag_system.index is not None:
                    st.success(f"✅ RAG index loaded ({len(rag_system.texts)} chunks)")
                else:
                    st.warning("⚠️ No RAG index found")
                    if st.button("Build Index"):
                        with st.spinner("Building index..."):
                            try:
                                rag_system.build_index(force_rebuild=True)
                                st.success("✅ Index built!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed: {e}")

# Main interface
st.markdown("### 💬 Ask me anything about our knowledge base")

# Input section
query = st.text_input(
    "Your question:",
    placeholder="What would you like to know?",
    help="I can help you find information from our enterprise knowledge base",
    key="query_input"
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    submit = st.button("🚀 Submit", type="primary", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear", use_container_width=True)
with col3:
    # Show remaining turns
    if conversation_manager.conversation:
        remaining = conversation_manager.get_turns_remaining()
        if remaining <= 3 and remaining > 0:
            st.caption(f"⏳ {remaining} left")

# Main processing logic
if submit and query:
    # Process all queries through the conversational agent
    with st.spinner("🤖 Thinking..."):
        result = handle_conversational_query(query)
    
    # Display results
    if result["success"]:
        # Main response container
        response_container = st.container()
        
        with response_container:
            # Display based on response type and approval
            if result["approved"]:
                # Use different styling based on conversation mode
                if result.get("intent") == "session_end":
                    st.warning(f"**Assistant:** {result['response']}")
                    if result.get("session_ended"):
                        st.info("🔄 Session has been automatically reset. You can start a new conversation now!")
                elif result.get("intent") in ["greeting", "help"]:
                    st.info(f"**Assistant:** {result['response']}")
                elif result.get("intent") == "chitchat":
                    st.success(f"**Assistant:** {result['response']}")
                else:
                    # Information responses
                    if result.get("context"):
                        st.success(f"**Assistant:** {result['response']}")
                    else:
                        # No context found
                        st.warning(f"**Assistant:** {result['response']}")
            else:
                # Response was rejected by guard
                st.error("⚠️ **Safety Filter Active**")
                st.warning(f"**Assistant:** {result['response']}")
                
                if debug_mode and "guard_reasoning" in result.get("debug_info", {}):
                    with st.expander("🛡️ Guard Details"):
                        st.write(f"**Reason:** {result['debug_info']['guard_reasoning']}")
        
        # Context display (only show if there's actual RAG content)
        if show_context and result.get("context") and len(result["context"]) > 0:
            with st.expander(f"📚 Retrieved Context ({len(result['context'])} chunks)"):
                for i, chunk in enumerate(result["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    if i < len(result["context"]) - 1:
                        st.divider()
        
        # Show follow-up suggestions if available
        if result.get("debug_info", {}).get("follow_up_suggestions"):
            with st.expander("💡 Suggestions"):
                for suggestion in result["debug_info"]["follow_up_suggestions"]:
                    st.write(f"• {suggestion}")
        
        # Debug information
        if debug_mode:
            with st.expander("🔧 Debug Information"):
                # Create safe debug info
                safe_debug = result.get("debug_info", {}).copy()
                
                # Add additional debug context
                safe_debug["intent"] = result.get("intent", "unknown")
                safe_debug["topic"] = result.get("topic", "none")
                safe_debug["has_rag_content"] = len(result.get("context", [])) > 0
                safe_debug["response_approved"] = result.get("approved", False)
                
                # Display as formatted JSON
                st.json(safe_debug)
    else:
        # Error handling
        st.error(f"❌ {result['response']}")
        
        if debug_mode and "debug_info" in result:
            with st.expander("🔧 Error Details"):
                st.json(result["debug_info"])

elif clear:
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.85em; padding: 10px;'>"
    "🛡️ <b>Enterprise Safe Assistant</b><br>"
    "I provide information exclusively from our knowledge base. "
    "When information isn't available, I'll let you know honestly and suggest alternatives."
    "</div>", 
    unsafe_allow_html=True
)