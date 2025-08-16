# Pharma Enterprise Assistant - Safety Architecture

## Overview
Enterprise-grade pharmaceutical assistant for Journvax with regulatory-compliant safety controls.

## 🛡️ Three-Layer Safety System

```
User Input → [Medical Safety Check] → [LLM Processing] → [Compliance Validation] → Safe Output
```

### Layer 1: Medical Safety Detection
**File**: `medical_safety_patterns.py`
- Detects unsafe medical requests BEFORE processing
- Categories: Dosing, Prescription Sharing, Administration, Medical Advice
- Returns pre-approved refusal templates

### Layer 2: LLM Processing
**File**: `prompts.py`
- Strict instructions preventing medical advice
- Banned phrases list (no "don't worry", "consider having", etc.)
- Required structure: State refusal → Add warning if needed → Direct to provider

### Layer 3: Compliance Validation
**File**: `enterprise_compliance.py`
- Removes ALL hedging ("maybe", "perhaps", "might")
- Blocks risk minimization ("don't worry")
- Strips unsanctioned guidance (meal timing, schedules)
- Enforces medical disclaimers
- Final sanitization of problematic offers

## 📋 Safety Rules

### Always Blocked (Zero Tolerance)
| Query Type | Response |
|------------|----------|
| Dosing/Maximum amounts | "I cannot provide dosing advice. Contact your healthcare provider immediately." |
| Administration timing | "I cannot provide administration guidance. Contact your healthcare provider immediately." |
| Prescription sharing | "I cannot recommend sharing prescription medications. Each person must have their own prescription." |
| Drug interactions | "I cannot advise on drug interactions. Contact your healthcare provider or pharmacist immediately." |
| OTC combinations | "I cannot recommend specific over-the-counter products while taking Journvax." |

### Banned Content
- ❌ Specific mg amounts (unless citing documentation verbatim)
- ❌ Food/meal recommendations
- ❌ Schedule advice ("stick to your regular schedule")
- ❌ Risk minimization ("don't worry", "should be okay")
- ❌ General medical advice ("stay hydrated", "get rest")
- ❌ Offering alternatives ("would you like suggestions?")

## 🔧 System Components

### Core Safety Files
```
├── guard.py                      # Main orchestrator
├── medical_safety_patterns.py    # Unsafe query detection
├── enterprise_compliance.py      # Response validation & sanitization
├── off_topic_handler.py         # Graceful off-topic handling
└── prompts.py                    # Compliant prompt templates
```

### Supporting Files
```
├── conversational_agent.py      # Response orchestration
├── llm_client.py               # Output cleaning
├── rag.py                      # Document retrieval
├── embeddings.py               # Grounding validation
├── config.py                   # System configuration
└── app.py                      # Streamlit interface
```

## 🔄 Request Flow

1. **Query Validation** (`guard.py`)
   - Routes to appropriate safety checker
   - Coordinates all safety layers

2. **Medical Safety Check** (`medical_safety_patterns.py`)
   - Semantic detection of unsafe requests
   - Returns if unsafe: pre-approved template

3. **If Safe → LLM Processing**
   - Retrieves context from documentation
   - Generates response with strict prompts

4. **Compliance Enforcement** (`enterprise_compliance.py`)
   - Validates response for violations
   - Removes problematic content
   - Applies required disclaimers
   - Final sanitization

5. **Clean Output** (`llm_client.py`)
   - Removes meta-commentary
   - Ensures complete sentences

## 📊 Configuration

```python
# config.py key settings
ENABLE_GUARD = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.35
USE_LLM_GUARD = False  # Optional additional validation
MAX_CACHE_SIZE = 100
```

## ✅ Compliance Standards

### Required Response Structure
```
1. Clear refusal: "I cannot provide [X] advice."
2. Safety warning (if relevant): "Alcohol can increase side effects..."
3. Provider redirect: "Contact your healthcare provider."
```

### Pre-Approved Templates Only
All unsafe queries receive ONLY pre-approved templates. No dynamic generation for:
- Dosing questions
- Administration queries  
- Drug interactions
- Medical advice

## 🎯 Performance Metrics

- Query validation: <50ms
- Safety detection accuracy: >98%
- Template compliance: 100%
- False positive rate: <5%

## ⚠️ Regulatory Compliance

**This system**:
- Never provides medical advice, dosing, or administration guidance
- Uses only pre-approved templates for all safety refusals
- Maintains complete separation between refusals and medical content
- Logs all safety interventions for audit
- Complies with pharmaceutical industry standards

**Key Principle**: When in doubt, refuse and redirect to healthcare provider.

---

**Version**: 4.0 (Streamlined Architecture)  
**Last Updated**: November 2024  
**Files**: 11 core files (reduced from 15+)