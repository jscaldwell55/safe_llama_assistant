# Safe Enterprise Assistant (Pharma) – README

## System Overview

### Core Safety Principles

- **100% Document Grounding** - Responses ONLY from retrieved documentation, no external knowledge
- **Zero Creative Content** - Blocks all stories, poems, fiction, roleplay requests
- **Comprehensive Threat Detection** - Blocks violence, illegal drugs, harmful content
- **Mandatory Validation** - Every response validated before delivery
- **Intelligent Query Classification** - Distinguishes factual information requests from personal medical advice

### Key Updates (Version 3.1)

- **Improved Query Classification** - Allows legitimate drug interaction information while blocking personal advice
- **Enhanced Output Cleaning** - Removes metadata, timestamps, and prompt artifacts
- **Adjusted Grounding Sensitivity** - Better balance between safety and information access
- **Fixed Drug Interaction Queries** - "What can I not take with Journvax?" now properly answered
- **Metadata Leakage Prevention** - Enhanced cleaning of document IDs and timestamps

## Safety Architecture

### 1. Query Pre-Screening

**Allowed Information Requests:**
- Drug interactions and contraindications ("What can I not take with Journvax?")
- Who can/cannot take the medication ("Who should not take Journvax?")
- Food and drink restrictions ("What foods should I avoid?")
- General safety information and side effects
- Dosing information from documentation

**Blocked Personal/Harmful Queries:**
- Personal medical advice ("Can I take this with my warfarin?")
- Violence/self-harm/suicide references
- Illegal drugs (cocaine, heroin, "speedball", etc.)
- Creative content requests (stories, poems, fiction)
- Off-topic queries (physics, weather, recipes)
- Off-label use for specific individuals

**Standard Responses:**
- Safety violations: `"I'm sorry, I cannot discuss that. Would you like to talk about something else?"`
- No information: `"I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"`

### 2. Document Grounding (STRICT)

- **Similarity Threshold:** 0.50
- **Grounding Check:** Mandatory for all generated responses
- **No Context = No Response:** Returns standard refusal if no documentation retrieved
- **Unsupported Claims Detection:** Balanced to avoid false positives on legitimate information

### 3. Response Generation

**Strict Prompting Rules:**
- ONLY use explicitly stated information from documentation
- NEVER add general knowledge or training data
- NEVER create narratives or stories
- No meta-commentary about the response process
- Always return standard refusal if information unavailable

### 4. Mandatory Validation

**Every Response Validated For:**
- Document grounding (similarity ≥ 0.50)
- Regulatory compliance (6 critical categories)
- Creative content detection
- Off-topic detection
- Violence/illegal content
- Metadata leakage

### 5. Critical Regulatory Categories

1. **Off-Label/Unapproved Use** - Unapproved populations or indications
2. **Medical Advice** - Personal medical recommendations (not general information)
3. **Cross-Product References** - Misleading brand associations
4. **Administration Misuse** - Dangerous administration methods
5. **Inaccurate Claims** - False or unsupported product claims
6. **Inadequate Risk Communication** - Missing safety warnings

## Example Interactions

| User Query | System Response | Category |
|------------|-----------------|----------|
| "what are the side effects?" | Lists side effects from documentation + disclaimer | ✅ Allowed |
| "who can take journvax?" | "Adults with moderate to severe acute pain..." | ✅ Allowed |
| "what can I not take with journvax?" | Lists drug/food interactions from documentation | ✅ Allowed |
| "can I take journvax with warfarin?" | "I'm sorry, I cannot discuss that..." | ❌ Personal advice |
| "can I take journvax with a speedball?" | "I'm sorry, I cannot discuss that..." | ❌ Illegal drug |
| "tell me a story about journvax" | "I'm sorry, I don't seem to have any information..." | ❌ Creative content |
| "tell me about gravity" | "I'm sorry, I don't seem to have any information..." | ❌ Off-topic |
| "my child has pain, can they take it?" | "I'm sorry, I cannot discuss that..." | ❌ Personal advice |

## File Structure

| File | Purpose | Version 3.1 Changes |
|------|---------|---------------------|
| `app.py` | Streamlit UI with chat interface | No changes |
| `guard.py` | Safety validation system | **UPDATE:** Better query classification, allows drug interaction info |
| `conversational_agent.py` | Response orchestrator | **UPDATE:** Improved debug info accuracy |
| `prompts.py` | Generation prompts | **UPDATE:** Added metadata prevention instructions |
| `llm_client.py` | Output cleaning | **MAJOR UPDATE:** Enhanced metadata/artifact removal |
| `rag.py` | FAISS vector search | No changes |
| `config.py` | Configuration | Maintained at SEMANTIC_SIMILARITY_THRESHOLD = 0.50 |
| Other files | Various utilities | Minor updates |

## Configuration

**Required `config.py` settings:**

```python
# Grounding threshold
SEMANTIC_SIMILARITY_THRESHOLD = 0.50

# Keep LLM guard disabled to prevent false positives
USE_LLM_GUARD = False

# Enable guard and caching
ENABLE_GUARD = True
ENABLE_RESPONSE_CACHE = True
```

## Known Issues & Limitations

### Current Issues Being Addressed

- **Output Artifacts** - Occasional metadata leakage (id: 551281 | timestamps)
- **Prompt Leakage** - May include "Please ask another question" or similar artifacts
- **Response Latency** - Generation can take 5-10 seconds for complex queries
- **Grounding Sensitivity** - Occasionally rejects legitimate information (tuning ongoing)

### System Limitations

- No streaming responses (all-at-once delivery)
- Single document source (PDFs only)
- No conversation memory in RAG retrieval
- Fixed embedding model (all-MiniLM-L6-v2)
- Context window limitations for very long documents



## Troubleshooting Guide

### "Legitimate information being blocked"
- Check `_find_unsupported_claims` sensitivity in `guard.py`
- Verify query isn't being misclassified as personal advice
- Review grounding threshold (may need adjustment for specific content)

### "Metadata appearing in responses"
- Update `clean_model_output` in `llm_client.py`
- Check for patterns: timestamps, IDs, pipe characters
- Verify PDFs don't contain embedded metadata

### "System blocking drug interaction questions"
- Ensure `guard.py` has legitimate information patterns
- Check that query classification happens before medical advice detection
- Verify latest version deployed

### "Responses include 'Please ask another question'"
- Add removal pattern to output cleaner
- Check prompt for instruction leakage
- Verify model parameters don't include unwanted stop sequences

## Security & Privacy

- No user data persistence beyond session
- PDFs processed locally only
- No external API calls except configured HF endpoint
- Memory-only cache, cleared on restart
- No browser storage (localStorage/sessionStorage) used
- No analytics or tracking

## Version History

### Version 3.1 (Current - August 2024)
- Fixed drug interaction information requests
- Improved personal advice vs. information classification
- Enhanced output cleaning for metadata/artifacts
- Adjusted grounding sensitivity to reduce false positives
- Better debug information accuracy

### Version 3.0
- Strict document grounding enforcement
- Comprehensive threat detection
- Creative content blocking
- Mandatory validation for all responses

### Version 2.0
- Removed LLM guard to prevent false positives
- Added response caching
- Initial output cleaning

### Version 1.0
- Initial release with basic RAG functionality

---

**Version**: 3.1 (Balanced Safety Edition)  
**Last Updated**: August 2024  
**Status**: Production-ready with ongoing refinements  
**License**: Proprietary  
**Support**: For issues, check troubleshooting guide and known issues first