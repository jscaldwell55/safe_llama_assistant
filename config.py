import os

# Hugging Face Configuration
try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    # Try to get the endpoint from secrets first, then environment
    HF_INFERENCE_ENDPOINT = st.secrets.get("HF_ENDPOINT", os.getenv("HF_ENDPOINT"))
except (ImportError, FileNotFoundError, AttributeError):
    HF_TOKEN = os.getenv("HF_TOKEN")  # Fallback to environment variable
    HF_INFERENCE_ENDPOINT = os.getenv("HF_ENDPOINT")

# Just validate that endpoint exists, don't reject specific URLs
if not HF_INFERENCE_ENDPOINT:
    print("WARNING: HF_ENDPOINT is not configured. Please set it in Streamlit secrets or environment variables.")
    print("Example format: https://[your-endpoint-id].endpoints.huggingface.cloud")
else:
    print(f"INFO: Using HF_ENDPOINT: {HF_INFERENCE_ENDPOINT[:50]}...")

# Model Configuration - Optimized for natural conversation
MODEL_PARAMS = {
    "max_new_tokens": 512,      # Increased for fuller responses
    "temperature": 0.7,         # Higher for more natural variation
    "do_sample": True,
    "top_p": 0.9,              # Higher for richer vocabulary
    "repetition_penalty": 1.1,  # Avoid repetitive phrasing
    "return_full_text": False   # Don't return the prompt in response
}

# RAG Configuration - Enhanced for better retrieval
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800              # Larger chunks for more context
CHUNK_OVERLAP = 150           # More overlap to preserve context
TOP_K_RETRIEVAL = 8           # More candidates for better coverage
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
EMBEDDING_BATCH_SIZE = 32     # Process embeddings in batches for better memory management

# Semantic Chunking Configuration
CHUNKING_STRATEGY = "hybrid"  # Can be: sections, paragraphs, sentences, recursive, hybrid
MAX_CHUNK_TOKENS = 800        # Maximum tokens per semantic chunk

# Guard Agent Configuration
ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # For grounding validation

# Conversation Configuration
MAX_CONVERSATION_TURNS = 10   # Reasonable conversation length
SESSION_TIMEOUT_MINUTES = 30   # Auto-end inactive sessions
MAX_CONTEXT_LENGTH = 4000     # Characters for context window

# UI Configuration
APP_TITLE = "🛡️ Safe Enterprise Assistant"
DEFAULT_FALLBACK_MESSAGE = "I don't have that information in our knowledge base. Could you rephrase your question or ask about something else?"

# System Messages - Simple and trust-based
SYSTEM_MESSAGES = {
    "no_context": "I don't have information about that in our documentation. Would you like to ask about something else?",
    "error": "I encountered an error processing your request. Please try again or start a new conversation.",
    "session_end": "We've reached the conversation limit. Thank you for chatting! Please start a new conversation to continue."
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'