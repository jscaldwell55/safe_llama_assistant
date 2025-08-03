import streamlit as st
import logging
from config import APP_TITLE
from rag import retrieve
from prompt import format_conversational_prompt
from llm_client import call_base_assistant, hf_client
from guard import evaluate_response
from conversation import conversation_manager
from context_formatter import context_formatter
from conversational_agent import conversational_agent, ConversationMode
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE, 
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title(APP_TITLE)

# Enhanced intent handling now handled by conversation_manager

def get_greeting_response():
    """Return greeting response"""
    return """👋 **Hello! I'm your enterprise knowledge assistant.**

I'm here to help you find information from our knowledge base. You can:

• Ask questions about our documentation
• Search for specific information or data
• Get insights from our enterprise content

What would you like to know today?"""

def get_help_response():
    """Return help information"""
    return """🤖 **How I can help you:**

• Ask questions about our enterprise knowledge base
• Get information from our documentation
• Find relevant data and insights

**Note:** I can only provide information based on the content in our knowledge base. For specific medical, legal, or financial advice, please consult appropriate professionals.

What would you like to know?"""

def handle_conversational_query(query: str):
    """
    Process a conversational query through enhanced conversational agent with RAG-only policy.
    
    Args:
        query (str): The user's question
        
    Returns:
        dict: Response information including final answer, context, and debug info
    """
    try:
        # Step 1: Use conversational agent to determine response strategy
        conv_response = conversational_agent.process_conversation(query)
        
        # Step 2: Handle session end specially
        if conv_response.mode == ConversationMode.SESSION_END:
            # Auto-reset conversation after session end
            conversation_manager.start_new_conversation()
            
            return {
                "success": True,
                "response": conv_response.text,
                "approved": True,
                "context": [],
                "intent": conv_response.mode.value,
                "topic": None,
                "session_ended": True,
                "debug_info": {
                    "conversation_mode": conv_response.mode.value,
                    "confidence": conv_response.confidence,
                    "follow_up_suggestions": conv_response.follow_up_suggestions,
                    "session_auto_reset": True
                }
            }
        
        # Step 3: Handle responses that don't need RAG content
        if not conv_response.has_rag_content:
            # Update conversation for social interactions
            if conv_response.mode in [ConversationMode.GREETING, ConversationMode.HELP]:
                conversation_manager.start_new_conversation()
            
            return {
                "success": True,
                "response": conv_response.text,
                "approved": True,  # Pre-approved conversational responses
                "context": [],
                "intent": conv_response.mode.value,
                "topic": None,
                "debug_info": {
                    "conversation_mode": conv_response.mode.value,
                    "confidence": conv_response.confidence,
                    "follow_up_suggestions": conv_response.follow_up_suggestions,
                    "rag_content_required": False
                }
            }
        
        # Step 4: For responses requiring RAG content, proceed with retrieval
        intent, topic = conversation_manager.classify_intent(query)
        conversation_context = conversation_manager.get_conversation_context()
        enhanced_query = conversation_manager.get_enhanced_query(query)
        
        logger.info(f"Processing RAG query: {query[:50]}... (mode: {conv_response.mode.value})")
        context_chunks = retrieve(enhanced_query)
        
        # Step 5: Strict RAG-only validation
        if not context_chunks or len(context_chunks) == 0:
            fallback_response = "I'm sorry, I don't seem to have any information on that. Can I help you with something else?"
            
            # Add conversational context for follow-ups
            if conv_response.mode == ConversationMode.FOLLOW_UP and conversation_context:
                fallback_response = "I don't have additional information on that topic in our knowledge base. Would you like to explore something else?"
            
            return {
                "success": False,
                "response": fallback_response,
                "approved": True,  # Pre-approved fallback
                "context": [],
                "intent": conv_response.mode.value,
                "topic": topic,
                "debug_info": {
                    "conversation_mode": conv_response.mode.value,
                    "error": "No RAG content found",
                    "enhanced_query": enhanced_query,
                    "rag_content_required": True
                }
            }
        
        # Step 6: Format context and generate response
        conversation_entities = conversation_manager.conversation.active_entities if conversation_manager.conversation else []
        formatted_context = context_formatter.format_enhanced_context(
            context_chunks, query, conversation_context, conversation_entities
        )
        
        # Step 7: Generate RAG-based response with conversational enhancement
        base_prompt = format_conversational_prompt(
            query, formatted_context, conversation_context, conv_response.mode.value, topic
        )
        assistant_response = call_base_assistant(base_prompt)
        
        # Step 8: Enhance with conversational elements
        enhanced_response = conversational_agent.enhance_response_with_conversational_elements(
            assistant_response, conv_response.mode, query
        )
        
        # Step 9: Guard evaluation with enhanced RAG-only checking
        context_text = "\n\n".join(context_chunks)
        is_approved, final_response, guard_reasoning = evaluate_response(
            context_text, query, enhanced_response
        )
        
        # Step 10: Update conversation history
        if is_approved:
            conversation_manager.add_turn(query, final_response, context_chunks, topic)
        
        return {
            "success": True,
            "response": final_response,
            "approved": is_approved,
            "context": context_chunks,
            "intent": conv_response.mode.value,
            "topic": topic,
            "debug_info": {
                "conversation_mode": conv_response.mode.value,
                "base_prompt": base_prompt,
                "assistant_response": assistant_response,
                "enhanced_response": enhanced_response,
                "guard_reasoning": guard_reasoning,
                "enhanced_query": enhanced_query,
                "conversation_context": conversation_context,
                "confidence": conv_response.confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "success": False,
            "response": "I'm sorry, there was an error processing your request.",
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
    if st.button("New Conversation"):
        conversation_manager.start_new_conversation()
        st.success("Started new conversation")
        st.rerun()
    
    # Show conversation stats with session limit info
    if conversation_manager.conversation:
        turn_count = len(conversation_manager.conversation.turns)
        turns_remaining = conversation_manager.get_turns_remaining()
        
        if turn_count > 0:
            if turns_remaining > 0:
                st.info(f"📊 Conversation: {turn_count}/{conversation_manager.max_turns} turns")
                if turns_remaining <= 2:
                    st.warning(f"⚠️ {turns_remaining} turns remaining before session ends")
            else:
                st.error("🔴 Session limit reached - next response will end session")
                
            if conversation_manager.conversation.current_topic:
                st.info(f"💬 Topic: {conversation_manager.conversation.current_topic}")
        else:
            st.info("💫 Ready to start conversation")
    
    # Health check
    with st.expander("System Status"):
        if st.button("Check System Health"):
            with st.spinner("Checking system health..."):
                # Check HF endpoint
                hf_healthy = hf_client.health_check()
                if hf_healthy:
                    st.success("✅ Hugging Face endpoint is healthy")
                else:
                    st.error("❌ Hugging Face endpoint is not responding")
                
                # Check RAG system
                from rag import rag_system
                if rag_system.index is not None:
                    st.success(f"✅ RAG index loaded ({len(rag_system.texts)} chunks)")
                else:
                    st.warning("⚠️ No RAG index found")
                    if st.button("Build Index from Sample Data"):
                        with st.spinner("Building index..."):
                            try:
                                from rag import build_index
                                build_index(force_rebuild=True)
                                st.success("✅ Index built successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to build index: {e}")

# Main interface
st.markdown("### Ask me anything about our knowledge base")

# Input section
query = st.text_input(
    "Enter your question:",
    placeholder="What would you like to know?",
    help="Ask questions about the content in our knowledge base"
)

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Submit", type="primary")
with col2:
    if st.button("Clear"):
        st.rerun()

# Main processing logic
if submit and query:
    # Process all queries through the conversational agent
    with st.spinner("🤖 Processing your question..."):
        result = handle_conversational_query(query)
    
    # Display results
    if result["success"]:
        # Main response
        if result["approved"]:
            # Use different styling based on conversation mode
            if result.get("intent") == "session_end":
                st.warning(f"**Assistant:** {result['response']}")
                if result.get("session_ended"):
                    st.info("🔄 Session has been automatically reset. You can start a new conversation now!")
            elif result.get("intent") in ["greeting", "help", "chitchat"]:
                st.info(f"**Assistant:** {result['response']}")
            else:
                st.success(f"**Assistant:** {result['response']}")
        else:
            st.error("⚠️ **Safety Filter Active**")
            st.warning(result['response'])
            
            if debug_mode:
                with st.expander("🛡️ Guard Details"):
                    st.write(f"**Reason:** {result['debug_info'].get('guard_reasoning', 'Unknown')}")
                    if 'assistant_response' in result['debug_info']:
                        st.write(f"**Original Response:** {result['debug_info']['assistant_response']}")
        
        # Context display (only show if there's actual RAG content)
        if show_context and result["context"] and len(result["context"]) > 0:
            with st.expander(f"📚 Retrieved Context ({len(result['context'])} chunks)"):
                for i, chunk in enumerate(result["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()
        
        # Show follow-up suggestions if available
        if "follow_up_suggestions" in result.get("debug_info", {}) and result["debug_info"]["follow_up_suggestions"]:
            with st.expander("💡 Suggestions"):
                for suggestion in result["debug_info"]["follow_up_suggestions"]:
                    st.write(f"• {suggestion}")
        
        # Debug information
        if debug_mode:
            with st.expander("🔧 Debug Information"):
                debug_info = result["debug_info"].copy()
                # Add conversation info
                debug_info["intent"] = result.get("intent", "unknown")
                debug_info["topic"] = result.get("topic", "none")
                debug_info["has_rag_content"] = len(result.get("context", [])) > 0
                st.json(debug_info)
    else:
        # Handle errors with conversational tone
        if "I don't seem to have any information" in result['response']:
            st.warning(f"**Assistant:** {result['response']}")
        else:
            st.error(f"❌ **Error:** {result['response']}")
        
        if debug_mode and "debug_info" in result:
            st.code(result["debug_info"])

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "🛡️ This assistant engages naturally in conversation while strictly using only information from our knowledge base. "
    "When information isn't available, I'll let you know honestly."
    "</div>", 
    unsafe_allow_html=True
)
