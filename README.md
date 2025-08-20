# Safe Enterprise Assistant (Pharma) — README

A production-ready pharmaceutical RAG assistant that answers questions only from approved documents (Medication Guide/PI) with strict safety enforcement. Built with Streamlit, FAISS, SentenceTransformers, and Hugging Face Inference Endpoints.

## System Overview

### Core Capabilities
- **Document-grounded responses only** - All answers come from uploaded pharmaceutical documentation
- **Automatic compliance corrections** - Adds required disclaimers without blocking legitimate information
- **Fast response caching** - Previously validated responses are cached for speed
- **Clean, professional output** - Advanced text cleaning removes meta-commentary and formatting issues

### Safety Architecture
- **2-Layer validation system**: Pattern-based rules + document grounding verification
- **No false positives**: LLM guard disabled to prevent over-blocking of legitimate pharmaceutical information
- **Smart auto-corrections**: Adds FDA-required disclaimers automatically instead of refusing

## What's Included (Files)

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI with chat interface and debug mode |
| `conversational_agent.py` | Main orchestrator handling retrieval → generation → validation → caching |
| `guard.py` | Safety validation system with pattern matching and grounding checks |
| `llm_client.py` | Async HF endpoint client with comprehensive output cleaning |
| `rag.py` | FAISS vector search, PDF parsing, context retrieval |
| `semantic_chunker.py` | Intelligent document chunking with section awareness |
| `context_formatter.py` | Deduplicates and compacts retrieved context |
| `embeddings.py` | SentenceTransformer singleton (all-MiniLM-L6-v2) |
| `conversation.py` | Session and conversation state management |
| `prompts.py` | Generation prompts with strict grounding requirements |
| `config.py` | All configuration: thresholds, model params, UI text |

## End-to-End Workflow

```
User Query
    │
    ▼
Query Pre-screening (guard.validate_query)
    ├─ Blocks: Unsafe requests (dosing advice, pediatric use, misuse)
    └─ Allows: Information requests about medication
    │
    ▼
Context Retrieval (rag.retrieve_and_format_context)
    ├─ FAISS similarity search (top-4 chunks)
    └─ Context formatting and deduplication
    │
    ▼
Response Generation (llm_client + enhanced prompt)
    ├─ Strict grounding to retrieved documentation
    └─ Proper spelling (Journvax) and formatting
    │
    ▼
Response Validation (guard.validate_response)
    ├─ Pattern compliance checks
    ├─ Auto-corrections (adds disclaimers)
    └─ Grounding verification (embedding similarity)
    │
    ▼
Output Cleaning (llm_client.clean_model_output)
    ├─ Removes meta-commentary
    ├─ Fixes incomplete sentences
    └─ Ensures professional formatting
    │
    ▼
Cache & Deliver
    └─ Approved responses cached for future use
```

## Current Safety Pipeline

### 1. Query Pre-Screening
**Deterministic pattern matching** blocks clearly unsafe requests:
- Personal dosing questions ("how much should I take?")
- Pediatric/off-label use ("can I give to my child?")
- Administration misuse ("can I crush/snort/inject?")
- Prescription sharing requests

### 2. Response Generation
**Strict grounding prompt** ensures:
- Only information from documentation is used
- No general medical knowledge added
- Proper medication name spelling (Journvax)
- No meta-commentary about compliance

### 3. Pattern-Based Validation
**Auto-corrections** (not refusals):
- **Side effects** → Adds "This is not a complete list. See the Medication Guide for full information."
- **Severe symptoms** → Adds "If you experience severe symptoms, seek immediate medical attention."

**Hard blocks** for:
- Implied safety from absence ("doesn't mention X, so you're fine")
- Inappropriate reassurance ("don't worry", "should be fine")
- Personal medical advice ("you should take/stop")
- Unsourced dosing information

### 4. Document Grounding Check
**Embedding similarity validation**:
- Model: all-MiniLM-L6-v2
- Threshold: 0.35 (allows paraphrasing)
- Fails only if: Low similarity AND multiple unsupported claims

### 5. Output Cleaning
**Comprehensive text processing**:
- Removes parenthetical self-editing ("(rephrased for clarity)")
- Cuts off re-answer attempts ("However, since you're asking...")
- Fixes incomplete sentences ending with ", it."
- Corrects misspellings (JOURNAVX → Journvax)
- Ensures proper punctuation and spacing

## Configuration

Key settings in `config.py`:

```python
# Guard Configuration
ENABLE_GUARD = True                    # Master switch for safety system
USE_LLM_GUARD = False                  # Disabled to prevent false positives
SEMANTIC_SIMILARITY_THRESHOLD = 0.35   # Grounding check threshold

# Model Parameters
BRIDGE_SYNTHESIZER_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 0.6,
    "stop": ["\nUser:", "\nHuman:", "###"]
}

# Caching
ENABLE_RESPONSE_CACHE = True
MAX_CACHE_SIZE = 100

# RAG Settings
TOP_K_RETRIEVAL = 4
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
```

## User Interface

### Main Chat
- Clean chat interface with message history
- Automatic response validation and correction
- Professional, grounded responses

### Sidebar Controls
- **🔄 New Conversation** - Start fresh conversation
- **🗑️ Clear Response Cache** - Clear cached responses
- **🔧 Debug Mode** - Shows timing, validation details, grounding scores

### Debug Information (when enabled)
```json
{
  "timing": {
    "rag_ms": 150,
    "generation_ms": 2500,
    "total_ms": 3200
  },
  "validation": {
    "result": "rejected",
    "was_corrected": true,
    "grounding_score": 0.78
  }
}
```

## Known Improvements from Original

### ✅ Fixed Issues
1. **False positives eliminated** - LLM guard disabled, preventing blocking of legitimate information
2. **Proper FDA language handling** - "See the Medication Guide" no longer flagged as violation
3. **Clean responses** - Meta-commentary and re-answers removed
4. **Correct spelling** - "Journvax" consistently spelled correctly
5. **Auto-corrections work** - Disclaimers added without blocking responses

### 🚀 Performance
- Response time: 2-5 seconds typical
- Cached responses: <100ms
- Grounding check: ~200ms
- Pattern validation: <50ms

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
export HF_TOKEN="your-huggingface-token"
export HF_INFERENCE_ENDPOINT="your-endpoint-url"
```

### 3. Add Documentation
Place PDF files in `./data/` directory

### 4. Build Index
```python
from rag import build_index
build_index(force_rebuild=True)
```

### 5. Run Application
```bash
streamlit run app.py
```

## Troubleshooting

### "Response still has issues after clearing cache"
- Restart the app completely to reload all singletons
- Check that `USE_LLM_GUARD = False` in config
- Verify the latest `llm_client.py` is deployed

### "Grounding failures on valid content"
- Check PDFs are properly loaded in `./data/`
- Rebuild index: `build_index(force_rebuild=True)`
- Verify embedding model loaded successfully

### "Incomplete sentences in responses"
- Update to latest `llm_client.py` with enhanced cleaning
- Check model's `max_new_tokens` isn't too low

## Safety Compliance

### ✅ What's Allowed
- Listing side effects from documentation
- Standard FDA disclaimers ("See Medication Guide")
- Directing to healthcare providers
- Factual information from PI/Medication Guide

### ❌ What's Blocked
- Personal medical advice
- Dosing recommendations
- Pediatric use when not approved
- Dangerous reassurances
- Ungrounded claims

## Architecture Notes

- **Singleton pattern**: Guard, RAG system, and conductor use singletons
- **Async throughout**: All LLM calls are async for better performance
- **Streaming disabled**: Responses generated completely before display
- **Session state**: Streamlit session state manages conversation history

## Current Limitations

1. **No streaming**: Responses appear all at once after generation
2. **Single document source**: Only uses uploaded PDFs, no external knowledge
3. **No conversation memory in RAG**: Each query treated independently for retrieval
4. **Fixed embedding model**: all-MiniLM-L6-v2 hardcoded

## Security Notes

- No user data logged beyond session
- No external API calls except to configured HF endpoint
- PDFs processed locally, not sent to external services
- Cache stored in memory only, cleared on restart

---

**Version**: 2.0 (Post LLM-guard removal)  
**Last Updated**: August 2024  
**Status**: Production-ready with known limitations