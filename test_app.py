#!/usr/bin/env python3
"""
Debug test app for isolating Streamlit hanging issues.
Run this to test basic functionality and progressively add imports.
"""

import streamlit as st
import logging
import sys
import time

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("=== Starting minimal test app ===")

# Basic Streamlit configuration
st.set_page_config(page_title="Test App - Debug Mode")
st.title("🔧 Test App - Debug Mode")

st.write("✅ If you see this, basic Streamlit is working!")

# Test controls
with st.sidebar:
    st.header("Test Controls")
    test_level = st.selectbox(
        "Test Level",
        ["Basic", "Config Import", "RAG Import", "LLM Import", "Full App"]
    )

# Progressive testing based on level
if test_level == "Basic":
    st.success("Basic Streamlit is working correctly!")
    st.info("Try the next level to test config import.")

elif test_level == "Config Import":
    try:
        logger.info("Testing config import...")
        start = time.time()
        from config import APP_TITLE
        elapsed = time.time() - start
        st.success(f"✅ Config imported successfully in {elapsed:.2f}s")
        st.write(f"App Title: {APP_TITLE}")
    except Exception as e:
        st.error(f"❌ Config import failed: {e}")
        logger.error(f"Config import error: {e}", exc_info=True)

elif test_level == "RAG Import":
    try:
        # First test config
        logger.info("Testing config import...")
        from config import APP_TITLE
        st.success("✅ Config import OK")
        
        # Then test RAG
        logger.info("Testing RAG import...")
        start = time.time()
        
        # Test importing just the module first
        import rag
        st.info("✅ RAG module imported")
        
        # Then test accessing the global instance
        logger.info("Accessing rag_system instance...")
        from rag import rag_system
        elapsed = time.time() - start
        
        st.success(f"✅ RAG system imported successfully in {elapsed:.2f}s")
        
        if rag_system.index is not None:
            st.write(f"Index loaded with {len(rag_system.texts)} chunks")
        else:
            st.warning("No index loaded")
            
    except Exception as e:
        st.error(f"❌ RAG import failed: {e}")
        logger.error(f"RAG import error: {e}", exc_info=True)

elif test_level == "LLM Import":
    try:
        # Test all previous imports
        logger.info("Testing previous imports...")
        from config import APP_TITLE
        st.success("✅ Config import OK")
        
        # Test LLM client
        logger.info("Testing LLM client import...")
        start = time.time()
        
        # Test importing module
        import llm_client
        st.info("✅ LLM client module imported")
        
        # Test accessing global instance
        from llm_client import hf_client
        elapsed = time.time() - start
        
        st.success(f"✅ LLM client imported successfully in {elapsed:.2f}s")
        
        # Test health check with button
        if st.button("Test HF Health Check"):
            with st.spinner("Checking HF endpoint..."):
                healthy = hf_client.health_check()
                if healthy:
                    st.success("✅ HF endpoint is healthy!")
                else:
                    st.error("❌ HF endpoint is not responding")
                    
    except Exception as e:
        st.error(f"❌ LLM client import failed: {e}")
        logger.error(f"LLM client import error: {e}", exc_info=True)

elif test_level == "Full App":
    try:
        logger.info("Testing full app imports...")
        
        # Import everything progressively with timing
        imports = [
            ("config", "from config import APP_TITLE"),
            ("prompt", "from prompt import format_conversational_prompt"),
            ("guard", "from guard import evaluate_response"),
            ("context_formatter", "from context_formatter import context_formatter"),
            ("ConversationMode", "from conversational_agent import ConversationMode"),
            ("rag", "from rag import rag_system"),
            ("conversation", "from conversation import conversation_manager"),
            ("conversational_agent", "from conversational_agent import conversational_agent"),
            ("llm_client", "from llm_client import call_base_assistant, hf_client"),
        ]
        
        failed = False
        total_time = 0
        
        for name, import_stmt in imports:
            try:
                start = time.time()
                logger.info(f"Importing {name}...")
                exec(import_stmt)
                elapsed = time.time() - start
                total_time += elapsed
                st.success(f"✅ {name} imported in {elapsed:.2f}s")
            except Exception as e:
                st.error(f"❌ Failed to import {name}: {e}")
                logger.error(f"Import error for {name}: {e}", exc_info=True)
                failed = True
                break
        
        if not failed:
            st.success(f"🎉 All imports successful! Total time: {total_time:.2f}s")
            st.info("The app should work normally now.")
            
            # Show system status
            with st.expander("System Status"):
                # Check each component
                exec("rag_system = rag_system")
                if rag_system.index is not None:
                    st.success(f"✅ RAG: {len(rag_system.texts)} chunks")
                else:
                    st.warning("⚠️ RAG: No index")
                
                exec("hf_client = hf_client")
                if st.button("Check HF Health"):
                    healthy = hf_client.health_check()
                    if healthy:
                        st.success("✅ HF: Healthy")
                    else:
                        st.error("❌ HF: Not responding")
                        
    except Exception as e:
        st.error(f"❌ Full app test failed: {e}")
        logger.error(f"Full app error: {e}", exc_info=True)

# Footer with instructions
st.markdown("---")
st.markdown("""
### Debug Instructions:
1. Start with "Basic" to ensure Streamlit works
2. Progress through each level to identify where hanging occurs
3. Check the terminal/logs for detailed error messages
4. The import that causes hanging will help identify the problem module
""")