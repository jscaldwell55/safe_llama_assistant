# config.py
import os

# Always use the public hub for downloads (NOT your inference endpoint)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HUGGINGFACE_HUB_URL"] = "https://huggingface.co"

# Hugging Face credentials / endpoint
try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    HF_INFERENCE_ENDPOINT = (
        st.secrets.get("HF_INFERENCE_ENDPOINT")
        or os.getenv("HF_INFERENCE_ENDPOINT")
        or os.getenv("HF_CUSTOM_ENDPOINT")
    )
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_INFERENCE_ENDPOINT = os.getenv("HF_INFERENCE_ENDPOINT") or os.getenv("HF_CUSTOM_ENDPOINT")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Provide it in Streamlit secrets or environment variables.")

if not HF_INFERENCE_ENDPOINT:
    raise RuntimeError(
        "HF_INFERENCE_ENDPOINT is not set. Example: https://<your-id>.endpoints.huggingface.cloud"
    )

print(f"INFO: Using HF_INFERENCE_ENDPOINT: {HF_INFERENCE_ENDPOINT[:50]}...")

# Model Generation defaults (conversational)
MODEL_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "return_full_text": False,
}

# RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 8
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
EMBEDDING_BATCH_SIZE = 32

# Semantic chunking
CHUNKING_STRATEGY = "hybrid"
MAX_CHUNK_TOKENS = 800

# Guard
ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.7

# Conversation
MAX_CONVERSATION_TURNS = 10         # 10 user↔assistant exchanges
SESSION_TIMEOUT_MINUTES = 30
MAX_CONTEXT_LENGTH = 4000           # characters

# UI
APP_TITLE = "🛡️ Safe Enterprise Assistant"
WELCOME_MESSAGE = "Hello! I can answer questions about our documentation. Ask me anything to get started."

DEFAULT_FALLBACK_MESSAGE = "I don't have that information in our knowledge base. Could you rephrase your question or ask about something else?"

SYSTEM_MESSAGES = {
    "no_context": "I don't have information about that in our documentation. Would you like to ask about something else?",
    "error": "I encountered an error processing your request. Please try again or start a new conversation.",
    "session_end": "We've reached the conversation limit. Thank you for chatting! Please start a new conversation to continue."
}

# Logging defaults for libraries (actual basicConfig only in app.py)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
