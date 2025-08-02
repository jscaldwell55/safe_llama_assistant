import os

# Hugging Face Configuration
try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
except ImportError:
    HF_TOKEN = os.getenv("HF_TOKEN")  # Fallback to environment variable
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://gqcc35s8tz0h3y0m.us-east-1.aws.endpoints.huggingface.cloud/")

# Model Configuration
MODEL_PARAMS = {
    "max_new_tokens": 300,
    "temperature": 0.3,
    "do_sample": True,
    "top_p": 0.9
}

# RAG Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"

# Guard Agent Configuration
GUARD_THRESHOLD = 0.7
ENABLE_GUARD = True

# UI Configuration
APP_TITLE = "🛡️ Safe Enterprise Assistant"
DEFAULT_FALLBACK_MESSAGE = "I'm sorry, I don't have that information in my knowledge base."
