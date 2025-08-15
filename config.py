# config.py

import os

# Ensure model downloads use the public hub (not your inference endpoint)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HUGGINGFACE_HUB_URL"] = "https://huggingface.co"

# Hugging Face Configuration
try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    HF_INFERENCE_ENDPOINT = st.secrets.get("HF_INFERENCE_ENDPOINT", os.getenv("HF_INFERENCE_ENDPOINT"))
    if not HF_INFERENCE_ENDPOINT:
        HF_INFERENCE_ENDPOINT = st.secrets.get("HF_ENDPOINT", os.getenv("HF_CUSTOM_ENDPOINT"))
except (ImportError, FileNotFoundError, AttributeError):
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_INFERENCE_ENDPOINT = os.getenv("HF_INFERENCE_ENDPOINT", os.getenv("HF_CUSTOM_ENDPOINT"))

if not HF_INFERENCE_ENDPOINT:
    print("WARNING: HF_INFERENCE_ENDPOINT is not configured. Set it in Streamlit secrets or env.")
    print("Example: https://<your-endpoint-id>.endpoints.huggingface.cloud")
else:
    print(f"INFO: Using HF_INFERENCE_ENDPOINT: {HF_INFERENCE_ENDPOINT[:50]}...")

# Model Generation Defaults
MODEL_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "return_full_text": False
}

# RAG Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 8
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
EMBEDDING_BATCH_SIZE = 32

# Semantic Chunking
CHUNKING_STRATEGY = "hybrid"
MAX_CHUNK_TOKENS = 800

# Guard Agent Configuration
ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.62  # numeric grounding threshold (per-claim)
LEXICAL_OVERLAP_MIN = 0.30           # lexical fallback for paraphrases
TERM_HIT_MIN = 3                      # enumeration leniency (≥3 exact term matches)
USE_LLM_GUARD = True                  # agentic LLM safety/grounding evaluator (cannot override numeric ungrounded gate)

# Conversation Configuration
MAX_CONVERSATION_TURNS = 0
SESSION_TIMEOUT_MINUTES = 30
MAX_CONTEXT_LENGTH = 4000

# UI Configuration
APP_TITLE = "Pharma Enterprise Assistant"
WELCOME_MESSAGE = "Hello! How can I help you today?"
DEFAULT_FALLBACK_MESSAGE = "I don't have that information in our knowledge base. Could you rephrase your question or ask about something else?"

# System Messages
SYSTEM_MESSAGES = {
    "no_context": "I don't have information about that in our documentation. Would you like to ask about something else?",
    "error": "I encountered an error processing your request. Please try again or start a new conversation.",
    "session_end": "We've reached the conversation limit. Thank you for chatting! Please start a new conversation to continue."
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
