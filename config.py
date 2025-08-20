# config.py - Simplified Configuration for Single Model System

import os

# Ensure downloads come from the public hub
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HUGGINGFACE_HUB_URL"] = "https://huggingface.co"

# ============================================================================
# HUGGING FACE CONFIGURATION
# ============================================================================

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
    print("WARNING: HF_INFERENCE_ENDPOINT is not configured.")
else:
    print(f"INFO: Using HF_INFERENCE_ENDPOINT: {HF_INFERENCE_ENDPOINT[:50]}...")

# ============================================================================
# MODEL PARAMETERS - SIMPLIFIED
# ============================================================================

# Base model parameters
MODEL_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "return_full_text": False,
    "use_cache": True,
}

# Bridge Synthesizer - Our main model
BRIDGE_SYNTHESIZER_PARAMS = {
    "max_new_tokens": 150,  # Reduced for faster responses
    "temperature": 0.6,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "return_full_text": False,
    # Only 4 stop sequences (HF limit)
    "stop": [
        "\nUser Question:",
        "\nUser:",
        "\nHuman:",
        "###"
    ]
}

# Guard model (if LLM guard is enabled)
GUARD_MODEL_PARAMS = {
    "max_new_tokens": 100,
    "temperature": 0.3,  # More deterministic for safety checks
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "return_full_text": False,
}

# ============================================================================
# CACHING
# ============================================================================

ENABLE_RESPONSE_CACHE = True
MAX_CACHE_SIZE = 100  # Maximum cached responses

# ============================================================================
# REQUEST BATCHING
# ============================================================================

# Disabled for now to reduce complexity
ENABLE_REQUEST_BATCHING = False
BATCH_TIMEOUT_MS = 50
MAX_BATCH_SIZE = 4

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 4
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
CHUNKING_STRATEGY = "hybrid"
MAX_CHUNK_TOKENS = 700
MAX_CONTEXT_LENGTH = 3500

# ============================================================================
# GUARD CONFIGURATION - SIMPLIFIED
# ============================================================================

ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.50  # Much lower - just needs some relationship
USE_LLM_GUARD = False  # Set to True for extra safety check
LLM_CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

MAX_CONVERSATION_TURNS = 0  # Unlimited
SESSION_TIMEOUT_MINUTES = 30

# ============================================================================
# UI CONFIGURATION
# ============================================================================

APP_TITLE = "Pharma Enterprise Assistant"
WELCOME_MESSAGE = "Hello! How can I help you today?"

DEFAULT_FALLBACK_MESSAGE = (
    "I don't have any information on that."
    "Could you rephrase your question or ask about something else?"
)

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_SLOW_REQUESTS_THRESHOLD_MS = 5000  # Log requests over 5s

