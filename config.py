# config.py - Simplified Configuration for Claude 3.5 Sonnet

import os

# ============================================================================
# ANTHROPIC/CLAUDE CONFIGURATION
# ============================================================================

try:
    import streamlit as st
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
except (ImportError, FileNotFoundError, AttributeError):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    print("WARNING: ANTHROPIC_API_KEY is not configured.")
else:
    print(f"INFO: Anthropic API key configured (length: {len(ANTHROPIC_API_KEY)})")

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Claude model configuration
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 500
TEMPERATURE = 0.3  # Lower for more consistent, grounded responses

# ============================================================================
# CACHING
# ============================================================================

ENABLE_RESPONSE_CACHE = True
MAX_CACHE_SIZE = 100  # Maximum cached responses

# ============================================================================
# RAG CONFIGURATION - Optimized for 100-200 pages
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
CHUNK_SIZE = 800  # Primary chunk size for RAG and semantic chunker
CHUNK_OVERLAP = 200  # More overlap for coherence
TOP_K_RETRIEVAL = 5  # Get top 5 most relevant chunks
INDEX_PATH = "faiss_index"
PDF_DATA_PATH = "data"
CHUNKING_STRATEGY = "hybrid"
MAX_CHUNK_TOKENS = CHUNK_SIZE # Aligning max tokens for semantic chunker with RAG chunk size
MAX_CONTEXT_LENGTH = 4000  # Can be larger with Claude

# ============================================================================
# GUARD CONFIGURATION - SIMPLIFIED
# ============================================================================

ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.45  # Adjusted for less strict grounding - was 0.60
GUARD_FALLBACK_MESSAGE = "I'm sorry, I can't discuss that. Can we talk about something else?"
NO_CONTEXT_FALLBACK_MESSAGE = "I'm sorry, I don't have any information on that. Can I assist you with something else?"

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

MAX_CONVERSATION_TURNS = 20  # Limit for context window management
SESSION_TIMEOUT_MINUTES = 30

# ============================================================================
# UI CONFIGURATION
# ============================================================================

APP_TITLE = "Pharma Enterprise Assistant"
WELCOME_MESSAGE = "Hello! I'm here to help you with information about Journvax. What would you like to know?"

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
LOG_SLOW_REQUESTS_THRESHOLD_MS = 3000  # Log requests over 3s