# Safe Enterprise Assistant (Pharma) – README

A production-ready pharmaceutical RAG assistant with **strict document grounding** and comprehensive safety enforcement. All responses come exclusively from approved pharmaceutical documentation with zero tolerance for external knowledge or creative content.

## System Overview

### Core Safety Principles
- **100% Document Grounding** - Responses ONLY from retrieved documentation, no external knowledge
- **Zero Creative Content** - Blocks all stories, poems, fiction, roleplay requests
- **Comprehensive Threat Detection** - Blocks violence, illegal drugs, harmful content
- **Mandatory Validation** - Every response validated before delivery
- **Standard Refusals** - Consistent safe responses for blocked content

### Key Updates (Version 3.0)
- **Enhanced Guard System** - Stricter grounding threshold (0.50), comprehensive violation detection
- **Illegal Drug Detection** - Blocks references to cocaine, heroin, "speedball", etc.
- **Creative Content Prevention** - Rejects any story/narrative generation attempts
- **Off-Topic Blocking** - Refuses non-Journvax queries (gravity, weather, etc.)
- **Mandatory Response Validation** - All responses validated, no bypasses

## Safety Architecture

### 1. Query Pre-Screening
**Immediate Blocks:**
- Violence/self-harm/suicide references
- Illegal drugs (including "speedball", cocaine, heroin, etc.)
- Creative content requests (stories, poems, fiction)
- Off-topic queries (physics, weather, recipes)
- Medical advice requests
- Off-label use inquiries

**Standard Response for Blocked Queries:**
```
"I'm sorry, I cannot discuss that. Would you like to talk about something else?"
```

### 2. Document Grounding (STRICT)
- **Similarity Threshold:** 0.50 (raised from 0.35)
- **Grounding Check:** Mandatory for all generated responses
- **No Context = No Response:** Returns standard refusal if no documentation retrieved
- **Unsupported Claims:** Any claim not in documentation triggers rejection

### 3. Response Generation
**Strict Prompting Rules:**
- ONLY use explicitly stated information from documentation
- NEVER add general knowledge
- NEVER create narratives or stories
- Always return standard refusal if information unavailable

### 4. Mandatory Validation
**Every Response Validated For:**
- Document grounding (similarity ≥ 0.50)
- Regulatory compliance (6 critical categories)
- Creative content detection
- Off-topic detection
- Violence/illegal content

### 5. Critical Regulatory Categories
1. **Off-Label/Unapproved Use** - Unapproved populations or indications
2. **Medical Advice** - Personal medical recommendations
3. **Cross-Product References** - Misleading brand associations
4. **Administration Misuse** - Dangerous administration methods
5. **Inaccurate Claims** - False or unsupported product claims
6. **Inadequate Risk Communication** - Missing safety warnings

## Standard System Responses

### For No Information Available:
```
"I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"
```

### For Safety Violations:
```
"I'm sorry, I cannot discuss that. Would you like to talk about something else?"
```

### For Greetings:
```
"Hello! How can I help you with information about Journvax today?"
```

## File Structure

| File | Purpose | Key Changes v3.0 |
|------|---------|-----------------|
| `app.py` | Streamlit UI with chat interface | No changes |
| `guard.py` | **Safety validation system** | **MAJOR UPDATE: Stricter grounding, illegal drug detection, creative content blocking** |
| `conversational_agent.py` | **Response orchestrator** | **UPDATE: Mandatory validation for ALL responses** |
| `prompts.py` | **Generation prompts** | **UPDATE: Strict grounding rules, no external knowledge** |
| `llm_client.py` | HF endpoint client | Optimized output cleaning |
| `rag.py` | FAISS vector search | No changes |
| `config.py` | **Configuration** | **UPDATE: SEMANTIC_SIMILARITY_THRESHOLD = 0.50** |
| `context_formatter.py` | Context deduplication | No changes |
| `semantic_chunker.py` | Document chunking | No changes |
| `embeddings.py` | Embedding model manager | No changes |
| `conversation.py` | Session management | No changes |

## Configuration Updates

**Required changes in `config.py`:**
```python
# Stricter grounding threshold
SEMANTIC_SIMILARITY_THRESHOLD = 0.50  # Raised from 0.35

# Keep LLM guard disabled to prevent false positives
USE_LLM_GUARD = False

# Other settings remain unchanged
ENABLE_GUARD = True
ENABLE_RESPONSE_CACHE = True
```

## End-to-End Workflow

```
User Query
    │
    ▼
Query Pre-screening (guard.validate_query)
    ├─ Blocks: Violence, illegal drugs, creative content, off-topic
    └─ Allows: Journvax information requests only
    │
    ▼
Context Retrieval (rag.retrieve_and_format_context)
    ├─ FAISS similarity search (top-4 chunks)
    └─ No context = automatic refusal
    │
    ▼
Response Generation (llm_client + strict prompt)
    ├─ ONLY from retrieved documentation
    └─ Standard refusal if no information
    │
    ▼
MANDATORY Response Validation (guard.validate_response)
    ├─ Grounding check (≥ 0.50 similarity)
    ├─ Regulatory compliance check
    ├─ Creative content detection
    └─ Off-topic detection
    │
    ▼
Output Cleaning (minimal)
    └─ Remove role markers, fix punctuation
    │
    ▼
Cache & Deliver
    └─ Only cache approved responses
```

## Example Interactions

| User Query | System Response |
|------------|----------------|
| "give me side effects" | Lists side effects from documentation + disclaimer |
| "can old people take journvax" | Information from documentation or standard refusal |
| "can i take journvax with a speedball" | "I'm sorry, I cannot discuss that..." |
| "tell me about journvax in a story" | "I'm sorry, I don't seem to have any information on that..." |
| "tell me about gravity" | "I'm sorry, I don't seem to have any information on that..." |
| "hello" | "Hello! How can I help you with information about Journvax today?" |

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

### 3. Update Configuration
Edit `config.py`:
```python
SEMANTIC_SIMILARITY_THRESHOLD = 0.50
USE_LLM_GUARD = False
```

### 4. Add Documentation
Place PDF files in `./data/` directory

### 5. Build Index
```python
from rag import build_index
build_index(force_rebuild=True)
```

### 6. Run Application
```bash
streamlit run app.py
```

## Safety Compliance

### ✅ What's Allowed
- Information directly from Journvax documentation
- Side effects listings with proper disclaimers
- Standard FDA warnings
- Factual medication information

### ❌ What's Blocked
- Creative content (stories, poems, narratives)
- Illegal drug references
- Violence or self-harm content
- Off-topic queries
- Medical advice
- Information not in documentation
- Speculation or external knowledge

## Troubleshooting

### "System accepting inappropriate queries"
- Verify `guard.py` is updated to v3.0
- Check `SEMANTIC_SIMILARITY_THRESHOLD = 0.50`
- Ensure `requires_validation = True` in all responses
- Clear cache and restart

### "Responses include external information"
- Check prompts in `prompts.py` for strict grounding rules
- Verify grounding threshold is 0.50
- Review retrieved context quality
- Rebuild FAISS index if needed

### "Creative content getting through"
- Verify creative content detection in `guard.py`
- Check prompt pre-screening logic
- Ensure validation is not bypassed

## Performance Metrics

- **Query screening:** <50ms
- **Context retrieval:** 150-300ms
- **Response generation:** 2-5 seconds
- **Validation:** 200-300ms
- **Cached responses:** <100ms

## Security & Privacy

- No user data logged beyond session
- PDFs processed locally
- No external API calls except HF endpoint
- Memory-only cache, cleared on restart
- No browser storage used

## Current Limitations

1. No streaming responses
2. Single document source (PDFs only)
3. No conversation memory in retrieval
4. Fixed embedding model (all-MiniLM-L6-v2)

## Version History

### Version 3.0 (Current)
- Strict document grounding enforcement
- Comprehensive threat detection
- Creative content blocking
- Mandatory validation for all responses
- Illegal drug detection

### Version 2.0
- Removed LLM guard to prevent false positives
- Added response caching
- Improved output cleaning

### Version 1.0
- Initial release with basic RAG functionality

---

**Version**: 3.0 (Strict Grounding Edition)  
**Last Updated**: August 2024  
**Status**: Production-ready with enhanced safety