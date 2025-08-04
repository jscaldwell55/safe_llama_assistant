"""
Async-optimized Streamlit application for the Safe LLaMA Assistant.
This version uses async operations for better performance.
"""

import asyncio
import streamlit as st
import logging
import time
from config import APP_TITLE, SESSION_TIMEOUT_MINUTES
from async_conversational_agent import AsyncConversationalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

async def get_or_create_agent():
    """Get existing agent or create a new one."""
    if st.session_state.agent is None:
        agent = AsyncConversationalAgent()
        await agent.initialize()
        st.session_state.agent = agent
    return st.session_state.agent

async def process_message(user_input: str):
    """Process user message asynchronously."""
    agent = await get_or_create_agent()
    
    # Create a context manager for the LLM client
    async with agent.llm_client or AsyncHuggingFaceClient() as llm_client:
        agent.llm_client = llm_client
        response = await agent.process_query(user_input)
    
    return response

def check_session_timeout():
    """Check if session has timed out."""
    if time.time() - st.session_state.last_activity > SESSION_TIMEOUT_MINUTES * 60:
        st.session_state.messages = []
        if st.session_state.agent:
            st.session_state.agent.reset_conversation()
        st.session_state.last_activity = time.time()
        return True
    return False

def main():
    """Main application function."""
    st.title(APP_TITLE)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Performance mode selector
        performance_mode = st.selectbox(
            "Performance Mode",
            ["Standard", "Batch Processing", "High Performance"],
            help="Select processing mode for queries"
        )
        
        # Session info
        st.divider()
        st.subheader("📊 Session Info")
        st.info(f"Messages: {len(st.session_state.messages)}")
        
        if st.session_state.agent:
            st.info(f"Conversation turns: {st.session_state.agent.turn_count}")
        
        # Clear conversation button
        if st.button("🔄 Clear Conversation", type="secondary"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.reset_conversation()
            st.rerun()
        
        # Rebuild index button
        if st.button("📚 Rebuild Index", type="secondary"):
            with st.spinner("Rebuilding index..."):
                async def rebuild():
                    agent = await get_or_create_agent()
                    await agent.rebuild_index()
                asyncio.run(rebuild())
            st.success("Index rebuilt successfully!")
    
    # Check session timeout
    if check_session_timeout():
        st.info("Session timed out. Starting a new conversation.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about our documentation..."):
        # Update last activity
        st.session_state.last_activity = time.time()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                # Process based on performance mode
                if performance_mode == "Batch Processing":
                    # In batch mode, we could process multiple queries if needed
                    response = asyncio.run(process_message(prompt))
                else:
                    # Standard async processing
                    response = asyncio.run(process_message(prompt))
                
                elapsed = time.time() - start_time
                
                # Display response
                st.markdown(response)
                
                # Show performance metrics in debug mode
                if st.checkbox("Show performance metrics", value=False, key=f"perf_{len(st.session_state.messages)}"):
                    st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer with performance stats
    with st.expander("🚀 Performance Optimizations"):
        st.markdown("""
        ### Enabled Optimizations:
        - **Batch Processing**: Embeddings processed in configurable batches (32-64)
        - **Async I/O**: All file operations and API calls are asynchronous
        - **Connection Pooling**: Reused connections for better performance
        - **Concurrent Processing**: Multiple operations run in parallel
        - **Smart Caching**: LRU cache for frequently accessed responses
        """)

if __name__ == "__main__":
    main()