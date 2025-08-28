# config.py - Configuration with Corrected Safety Thresholds

import os

# Ensure downloads come from the public hub
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HUGGINGFACE_HUB_URL"] = "https://huggingface.co"

# ============================================================================
# API CONFIGURATION
# ============================================================================

try:
    import streamlit as st
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
except (ImportError, FileNotFoundError, AttributeError):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 1000
TEMPERATURE = 0.4  # Lower for consistency

# ============================================================================
# SAFETY THRESHOLDS - CORRECTED FOR PRODUCTION
# ============================================================================

# CRITICAL: Raised from dangerous 0.25 to safe 0.75
SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # Minimum cosine similarity for grounding validation

# Retrieval quality threshold
MIN_RETRIEVAL_SCORE = 0.70  # Raised from 0.45 to ensure quality chunks

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 8
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
CHUNKING_STRATEGY = "hybrid"
MAX_CHUNK_TOKENS = 700
MAX_CONTEXT_LENGTH = 4000

# ============================================================================
# GUARD CONFIGURATION
# ============================================================================

ENABLE_GUARD = True
USE_LLM_GUARD = False
LLM_CONFIDENCE_THRESHOLD = 0.8

# Personal medical advice detection
BLOCK_PERSONAL_MEDICAL = True
PERSONAL_INDICATORS = ['my', 'i have', 'should i', 'can i', 'my grandmother', 
                       'my child', 'my mother', 'my father', 'my wife', 'my husband']
MEDICAL_CONTEXTS = ['take', 'use', 'safe', 'medication', 'journvax', 'dose', 'prescribe']

# ============================================================================
# CACHING
# ============================================================================

ENABLE_RESPONSE_CACHE = True
MAX_CACHE_SIZE = 100

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

MAX_CONVERSATION_TURNS = 20
SESSION_TIMEOUT_MINUTES = 30

# ============================================================================
# UI CONFIGURATION
# ============================================================================

APP_TITLE = "Pharma Enterprise Assistant"
WELCOME_MESSAGE = "Hello! I can help you with information about Journvax based on our documentation."

# Updated to be more accurate about capabilities
NO_CONTEXT_FALLBACK_MESSAGE = (
    "I don't have sufficient information in the documentation to answer that question. "
    "Please rephrase or ask about something else related to Journvax."
)

PERSONAL_MEDICAL_ADVICE_MESSAGE = (
    "I cannot provide personal medical advice. Please consult with a healthcare provider "
    "about whether Journvax is appropriate for specific individuals."
)

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_SLOW_REQUESTS_THRESHOLD_MS = 5000