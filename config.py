# config.py - A10G Optimized Configuration

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
# A10G OPTIMIZED MODEL PARAMETERS
# ============================================================================

# Base model parameters - optimized for A10G
MODEL_PARAMS = {
    "max_new_tokens": 256,  # Reduced from 512
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,  # Added for better quality
    "repetition_penalty": 1.1,
    "return_full_text": False,
    # A10G can handle larger batches
    "batch_size": 4,
    "use_cache": True,
}

# ============================================================================
# PERSONA-SPECIFIC OPTIMIZATIONS
# ============================================================================

# Intent Classifier - FAST (should take <1s on A10G)
INTENT_CLASSIFIER_PARAMS = {
    "max_new_tokens": 50,  # Very short - just need classification
    "temperature": 0.3,  # More deterministic
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "return_full_text": False,
}

# Empathetic Companion - BALANCED
EMPATHETIC_COMPANION_PARAMS = {
    "max_new_tokens": 120,  # Brief but warm
    "temperature": 0.8,  # More creative for empathy
    "do_sample": True,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "return_full_text": False,
}

# Information Navigator - PRECISE
INFORMATION_NAVIGATOR_PARAMS = {
    "max_new_tokens": 200,  # Enough for facts
    "temperature": 0.4,  # Lower for accuracy
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "return_full_text": False,
}

# Bridge Synthesizer - CREATIVE
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

# Guard Agent - FAST & PRECISE
GUARD_MODEL_PARAMS = {
    "max_new_tokens": 100,  # Reduced from 200
    "temperature": 0.3,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "return_full_text": False,
}

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

# Caching Configuration
ENABLE_RESPONSE_CACHE = True
CACHE_TTL_SECONDS = 3600  # 1 hour cache for common responses
MAX_CACHE_SIZE = 100  # Maximum cached responses

# Parallel Processing
ENABLE_PARALLEL_PERSONAS = True  # Run personas in parallel when possible
PARALLEL_TIMEOUT_SECONDS = 10  # Max wait for parallel operations

# Request Batching (for A10G efficiency)
ENABLE_REQUEST_BATCHING = False
BATCH_TIMEOUT_MS = 50  # Wait up to 50ms to batch requests
MAX_BATCH_SIZE = 4  # A10G can handle 4 concurrent requests efficiently

# Streaming Configuration
ENABLE_STREAMING = True
STREAM_CHUNK_SIZE = 10  # Tokens per chunk
STREAM_TIMEOUT_SECONDS = 30

# ============================================================================
# RAG OPTIMIZATION
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64  # Increased for A10G
CHUNK_SIZE = 600  # Slightly smaller for faster processing
CHUNK_OVERLAP = 100  # Reduced overlap
TOP_K_RETRIEVAL = 4  # Reduced from 6 for speed
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
CHUNKING_STRATEGY = "hybrid"
MAX_CHUNK_TOKENS = 700  # Reduced from 800

# RAG Caching
ENABLE_RAG_CACHE = True
RAG_CACHE_SIZE = 50
RAG_CACHE_TTL = 1800  # 30 minutes

# ============================================================================
# GUARD OPTIMIZATION
# ============================================================================

ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.60  # Slightly lower for speed
USE_LLM_GUARD = True
LLM_CONFIDENCE_THRESHOLD = 0.7

# Skip LLM guard for obviously safe responses
SKIP_GUARD_PATTERNS = [
    "hello", "hi", "thank you", "thanks", "goodbye", "bye",
    "good morning", "good afternoon", "good evening"
]

# Fast guard mode - skip expensive checks for common patterns
ENABLE_FAST_GUARD = True
FAST_GUARD_CONFIDENCE_THRESHOLD = 0.85

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

MAX_CONVERSATION_TURNS = 0  # Unlimited
SESSION_TIMEOUT_MINUTES = 30
MAX_CONTEXT_LENGTH = 3500  # Reduced from 4000 for speed

# Conversation caching
CACHE_CONVERSATION_HISTORY = True
CONVERSATION_CACHE_SIZE = 20

# ============================================================================
# UI CONFIGURATION
# ============================================================================

APP_TITLE = "Pharma Enterprise Assistant"
WELCOME_MESSAGE = "Hello! How can I help you today?"

DEFAULT_FALLBACK_MESSAGE = (
    "I don't have that information in our knowledge base. "
    "Could you rephrase your question or ask about something else?"
)

SYSTEM_MESSAGES = {
    "no_context": "I don't have information about that in the documentation. Would you like to ask about something else?",
    "error": "I encountered an error processing your request. Please try again or start a new conversation.",
    "session_end": "We've reached the conversation limit. Thank you for chatting! Please start a new conversation to continue."
}

# ============================================================================
# LATENCY TARGETS (A10G)
# ============================================================================

TARGET_LATENCIES = {
    "intent_classification": 1000,  # 1s
    "pure_empathy": 2000,  # 2s
    "pure_facts": 3000,  # 3s
    "synthesized": 5000,  # 5s
    "guard_validation": 500,  # 0.5s
}

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
ENABLE_PERFORMANCE_LOGGING = True
LOG_SLOW_REQUESTS_THRESHOLD_MS = 5000  # Log requests over 5s

# ============================================================================
# DEBUG MODE
# ============================================================================

SHOW_GUARD_REASONING = os.getenv("SHOW_GUARD_REASONING", "false").lower() == "true"
SHOW_LATENCY_BREAKDOWN = True  # Show detailed timing in debug mode
ENABLE_PROFILING = False  # Set True for detailed performance profiling