import streamlit as st
import logging
from config import APP_TITLE
from rag import retrieve
from prompt import format_conversational_prompt
from llm_client import call_base_assistant, hf_client
from guard import evaluate_response
from conversation import conversation_manager
from context_formatter import context_formatter
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
    Process a conversational query through enhanced RAG + Guard pipeline.
    
    Args:
        query (str): The user's question
        
    Returns:
        dict: Response information including final answer, context, and debug info
    """
    try:
        # Step 1: Classify intent and get conversation context
        intent, topic = conversation_manager.classify_intent(query)
        conversation_context = conversation_manager.get_conversation_context()
        
        # Step 2: Enhance query for better retrieval
        enhanced_query = conversation_manager.get_enhanced_query(query)
        
        # Step 3: Retrieve relevant context
        logger.info(f"Processing query: {query[:50]}... (intent: {intent})")
        context_chunks = retrieve(enhanced_query)
        
        if not context_chunks:
            # Handle no context found
            no_context_response = "I'm sorry, I couldn't find relevant information about that in my knowledge base."
            if conversation_context:
                no_context_response += " Could you clarify or ask about something else we've discussed?"
            
            return {
                "success": False,
                "response": no_context_response,
                "context": [],
                "debug_info": {"error": "No context retrieved", "intent": intent, "topic": topic}
            }
        
        # Step 4: Format enhanced context
        conversation_entities = conversation_manager.conversation.active_entities if conversation_manager.conversation else []
        formatted_context = context_formatter.format_enhanced_context(
            context_chunks, query, conversation_context, conversation_entities
        )
        
        # Step 5: Generate conversational response
        base_prompt = format_conversational_prompt(
            query, formatted_context, conversation_context, intent, topic
        )
        assistant_response = call_base_assistant(base_prompt)
        
        # Step 6: Guard evaluation
        context_text = "\n\n".join(context_chunks)
        is_approved, final_response, guard_reasoning = evaluate_response(
            context_text, query, assistant_response
        )
        
        # Step 7: Update conversation history
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
                "base_prompt": base_prompt,
                "assistant_response": assistant_response,
                "guard_reasoning": guard_reasoning,
                "enhanced_query": enhanced_query,
                "conversation_context": conversation_context
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
    
    # Show conversation stats
    if conversation_manager.conversation:
        turn_count = len(conversation_manager.conversation.turns)
        if turn_count > 0:
            st.info(f"Current conversation: {turn_count} turns")
            if conversation_manager.conversation.current_topic:
                st.info(f"Topic: {conversation_manager.conversation.current_topic}")
    
    # Health check
    with st.expander("System Status"):
        if st.button("Check System Health"):
            with st.spinner("Checking system health..."):
                hf_healthy = hf_client.health_check()
                
                if hf_healthy:
                    st.success("✅ Hugging Face endpoint is healthy")
                else:
                    st.error("❌ Hugging Face endpoint is not responding")

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
    # Get intent from conversation manager
    intent, topic = conversation_manager.classify_intent(query)
    
    if intent == "greeting":
        st.info(get_greeting_response())
        # Start fresh conversation on greeting
        conversation_manager.start_new_conversation()
    elif intent == "help_request":
        st.info(get_help_response())
    else:
        # Process conversational query
        with st.spinner("🤖 Processing your question..."):
            result = handle_conversational_query(query)
        
        # Display results
        if result["success"]:
            # Main response
            if result["approved"]:
                st.success(f"**Assistant:** {result['response']}")
            else:
                st.error("⚠️ **Safety Filter Active**")
                st.warning(result['response'])
                
                if debug_mode:
                    with st.expander("🛡️ Guard Details"):
                        st.write(f"**Reason:** {result['debug_info']['guard_reasoning']}")
                        st.write(f"**Original Response:** {result['debug_info']['assistant_response']}")
            
            # Context display
            if show_context and result["context"]:
                with st.expander(f"📚 Retrieved Context ({len(result['context'])} chunks)"):
                    for i, chunk in enumerate(result["context"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.divider()
            
            # Debug information
            if debug_mode:
                with st.expander("🔧 Debug Information"):
                    debug_info = result["debug_info"].copy()
                    # Add conversation info
                    debug_info["intent"] = result.get("intent", "unknown")
                    debug_info["topic"] = result.get("topic", "none")
                    st.json(debug_info)
        else:
            st.error(f"❌ **Error:** {result['response']}")
            if debug_mode:
                st.code(result["debug_info"])

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "🛡️ This assistant is powered by enterprise-grade safety filters and only responds with information from our knowledge base."
    "</div>", 
    unsafe_allow_html=True
)
