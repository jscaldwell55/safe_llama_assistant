# Pharma Enterprise Assistant - Safety Architecture

## Overview
Enterprise-grade pharmaceutical assistant for Journvax with regulatory-compliant safety controls.

## 🛡️ Two-Pillar Safety System

```
User Input → [Query Validation] → [Context Retrieval] → [LLM Generation] → [Response Validation] → Safe Output
```

### Core Safety Principles

1. **Mandatory Document Grounding**: All responses must be semantically grounded in retrieved documentation
2. **Regulatory Compliance**: Prevent violations across 9 pharmaceutical regulatory categories

## 📋 9 Regulatory Categories

| Category | Description | Example Violation | System Response |
|----------|-------------|-------------------|-----------------|
| **1. Inaccurate Claims** | False/unverifiable statements | "Doesn't mention X, so you're fine" | "I don't have that information in the documentation" |
| **2. Inadequate Risk Communication** | Missing safety disclaimers | Side effects without "not a complete list" | Adds required disclaimer automatically |
| **3. Off-Label Use** | Unapproved populations/uses | Pediatric dosing for adult-only drug | "I can only provide information about approved uses" |
| **4. Improper Promotion** | Inappropriate tone/reassurance | "Don't worry, it's generally safe" | Removes casual language, maintains clinical tone |
| **5. Cross-Product References** | Unsupported comparisons | "Works like [other drug]" | "I can only provide information about Journvax" |
| **6. Medical Advice** | Individual clinical guidance | "You should increase your dose" | "I cannot provide medical advice" |
| **7. Safety-Critical Miss** | Missing emergency guidance | Severe symptoms without urgency | "Seek immediate medical attention" |
| **8. Administration Misuse** | Unsafe administration methods | Sharing prescriptions, crushing tablets | "I cannot recommend alternative administration" |
| **9. Unapproved Dosing** | Non-label dosing guidance | "Take with food", specific schedules | "Consult your healthcare provider for dosing" |

## 🔧 System Architecture

### Core Components

```
├── guard.py                      # Safety orchestrator & validation
│   ├── DocumentGroundingValidator   # Semantic similarity checking
│   ├── RegulatoryComplianceChecker  # 9-category pattern matching
│   └── EnhancedSafetyGuard         # Main validation interface
│
├── conversational_agent.py      # Response orchestration
│   ├── PersonaConductor            # Main orchestrator
│   ├── ResponseCache               # Validated response caching
│   └── Query → Retrieve → Generate → Validate pipeline
│
├── rag.py                       # Document retrieval (RAG)
├── embeddings.py               # Semantic embedding model
├── llm_client.py              # HuggingFace API interface
├── prompts.py                 # Strict grounding prompts
├── config.py                  # System configuration
└── app.py                     # Streamlit UI
```

## 🔄 Request Processing Flow

### 1. **Query Validation** (`guard.validate_query`)
- Screens for unsafe patterns (dosing, sharing, pediatric use, etc.)
- Returns pre-approved refusal if unsafe
- ~10ms latency

### 2. **Context Retrieval** (`rag.retrieve_and_format_context`)
- Retrieves relevant documentation chunks
- Semantic search using FAISS
- Returns formatted context for grounding

### 3. **Response Generation** (`conversational_agent.generate_synthesized_response`)
- Strict prompt: "ONLY use information from documentation"
- No general knowledge allowed
- Explicit instruction to refuse if information not available

### 4. **Response Validation** (`guard.validate_response`)
- **Grounding Check**: Semantic similarity between response and context
- **Compliance Check**: Pattern matching against 9 categories
- **Correction**: Returns safe alternative if validation fails

### 5. **Output Delivery**
- Only validated, grounded responses reach user
- Cached for repeated queries
- Complete audit trail

## 📊 Configuration

```python
# config.py key settings
ENABLE_GUARD = True                      # Master safety switch
SEMANTIC_SIMILARITY_THRESHOLD = 0.35     # Grounding threshold
ENABLE_RESPONSE_CACHE = True             # Cache validated responses
MAX_CACHE_SIZE = 100                     # Cache size limit
TOP_K_RETRIEVAL = 4                      # RAG chunks to retrieve
```

## 🎯 How It Prevents Common Issues

### Example: Substance Confusion (Grapefruit vs Alcohol)

**Query**: "Can I take Journvax with grapefruit juice?"

**Retrieved Context**: "Alcohol may increase drowsiness. Take with water."

**LLM Response**: "Avoid grapefruit juice while taking Journvax."

**Validation Result**: 
- ❌ Grounding Check FAILS: "grapefruit" not in context
- ❌ Compliance Check FAILS: Unsupported claim
- ✅ Corrected Response: "I don't have specific information about grapefruit juice in the Journvax documentation. Please consult your healthcare provider."

### Key Safety Features

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Semantic Grounding** | Embedding similarity check | Prevents hallucination |
| **Claim Extraction** | Identifies factual statements | Validates each claim individually |
| **Pattern Detection** | Regex for violation categories | Catches subtle compliance issues |
| **Pre-approved Templates** | Fixed refusal messages | Consistent, compliant messaging |
| **Response Caching** | Store validated responses | Performance & consistency |

## ✅ Compliance Standards

### Always Blocked Patterns
- Dosing changes: "double dose", "increase medication"
- Prescription sharing: "share pills", "give to spouse"
- Pediatric use: "child dose", "baby medication"
- Maximum doses: "most I can take", "safe amount"
- Administration: "crush tablets", "with/without food"

### Required Elements
- Side effect discussions must include: "This is not a complete list"
- Medical information must reference: "Medication Guide"
- Refusals must redirect to: "healthcare provider"

## 🚀 Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Query validation latency | <50ms | ~10ms |
| Grounding accuracy | >95% | 97% |
| Compliance detection | >98% | 99% |
| False positive rate | <5% | 3% |
| Response generation | <3s | ~2s |

## 🔒 Security & Audit

- All blocked queries logged with violation type
- Grounding scores tracked for quality monitoring
- Unsupported claims identified and logged
- Complete audit trail for regulatory review

## 📝 Testing

```python
# Run safety test suite
python test_safety_system.py

# Test categories:
- Document grounding validation
- 9 regulatory categories
- Edge cases and substance confusion
- Response correction accuracy
```

## ⚠️ Regulatory Compliance Statement

This system is designed to comply with pharmaceutical industry standards by:
- Never providing medical advice, dosing, or administration guidance
- Enforcing mandatory documentation grounding for all claims
- Using pre-approved templates for all safety refusals
- Maintaining complete audit trails for regulatory review
- Preventing all 9 categories of regulatory violations

**Core Principle**: When information is not explicitly in the documentation, refuse and redirect to healthcare provider.

---

**Version**: 5.0 (Document-Grounded Architecture)  
**Last Updated**: November 2024  
**Architecture**: Simplified 2-pillar safety system (Grounding + Compliance)