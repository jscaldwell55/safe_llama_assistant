# config.py

import os

# Ensure downloads come from the public hub (not your inference endpoint)
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
    print("WARNING: HF_INFERENCE_ENDPOINT is not configured. Set it in secrets or environment.")
else:
    print(f"INFO: Using HF_INFERENCE_ENDPOINT: {HF_INFERENCE_ENDPOINT[:50]}...")

# ---------------- Model defaults ----------------
MODEL_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "return_full_text": False
}

# Conservative per-call cap for base assistant (latency)
BASE_MAX_NEW_TOKENS = 220

# Guard LLM specific parameters (lower temperature for more consistent evaluation)
GUARD_MODEL_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0.3,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "return_full_text": False
}

# ---------------- RAG ----------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K_RETRIEVAL = 6
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
EMBEDDING_BATCH_SIZE = 32
CHUNKING_STRATEGY = "hybrid"
MAX_CHUNK_TOKENS = 800

# ---------------- Guard ----------------
ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.62
# New: Control whether to use LLM evaluation in addition to heuristics
USE_LLM_GUARD = True
# Confidence threshold for LLM verdicts
LLM_CONFIDENCE_THRESHOLD = 0.7

# ---------------- Conversation ----------------
MAX_CONVERSATION_TURNS = 0      # 0 => unlimited
SESSION_TIMEOUT_MINUTES = 30
MAX_CONTEXT_LENGTH = 4000

# ---------------- UI ----------------
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

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ---------------- Debug Mode ----------------
# Show detailed guard reasoning in UI when enabled
SHOW_GUARD_REASONING = os.getenv("SHOW_GUARD_REASONING", "false").lower() == "true"