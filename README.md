Safe Enterprise Assistant (Pharma) – README
System Overview
Core Safety Principles

100% Document Grounding – Responses ONLY from retrieved documentation, no external knowledge


Comprehensive Threat Detection – Blocks violence, illegal drugs, harmful content

Mandatory Validation – Every response validated before delivery

Intelligent Query Classification – Distinguishes factual information requests from personal medical advice

Key Updates (Version 3.2)

Sentence-Level Grounding – Every generated sentence must meet the grounding threshold (0.50) individually

Chunk Sanitizer – Automatically strips PII, metadata artifacts, timestamps, and irrelevant boilerplate during RAG preprocessing

Improved Query Classification – Allows legitimate drug interaction information while blocking personal advice

Enhanced Output Cleaning – Removes metadata, timestamps, and prompt artifacts

Adjusted Grounding Sensitivity – Better balance between safety and information access

Fixed Drug Interaction Queries – "What can I not take with Journvax?" now properly answered

Metadata Leakage Prevention – Enhanced cleaning of document IDs and timestamps

Safety Architecture
1. Query Pre-Screening

Allowed Information Requests:

Drug interactions and contraindications

Who can/cannot take the medication

Food and drink restrictions

General safety information and side effects

Dosing information from documentation

Blocked Personal/Harmful Queries:

Personal medical advice

Violence/self-harm/suicide references

Illegal drugs (cocaine, heroin, “speedball”, etc.)

Creative content requests (stories, poems, fiction)

Off-topic queries (physics, weather, recipes)

Off-label use for specific individuals

Standard Responses:

Safety violations: "I'm sorry, I cannot discuss that. Would you like to talk about something else?"

No information: "I'm sorry, I don't seem to have any information on that. Would you like to talk about something else?"

2. Document Grounding (STRICT)

Similarity Threshold: 0.50

Chunk Sanitizer: Pre-cleans chunks before embedding, removing irrelevant metadata, timestamps, and document artifacts

Sentence-Level Grounding: Each sentence in the response is validated against retrieved chunks; ungrounded sentences are automatically dropped

No Context = No Response: Returns standard refusal if no documentation retrieved

Unsupported Claims Detection: Balanced to avoid false positives on legitimate information

3. Response Generation

Strict Prompting Rules:

ONLY use explicitly stated information from documentation

NEVER add general knowledge or training data

NEVER create narratives or stories

No meta-commentary about the response process

Always return standard refusal if information unavailable

4. Mandatory Validation

Every Response Validated For:

Sentence-level document grounding (≥ 0.50)

Regulatory compliance (6 critical categories)

Creative content detection

Off-topic detection

Violence/illegal content

Metadata leakage

5. Critical Regulatory Categories

Off-Label/Unapproved Use – Unapproved populations or indications

Medical Advice – Personal medical recommendations (not general information)

Cross-Product References – Misleading brand associations

Administration Misuse – Dangerous administration methods

Inaccurate Claims – False or unsupported product claims

Inadequate Risk Communication – Missing safety warnings

Example Interactions
User Query	System Response	Category
"what are the side effects?"	Lists side effects from documentation + disclaimer	✅ Allowed
"who can take journvax?"	"Adults with moderate to severe acute pain..."	✅ Allowed
"what can I not take with journvax?"	Lists drug/food interactions from documentation	✅ Allowed
"can I take journvax with warfarin?"	"I'm sorry, I cannot discuss that..."	❌ Personal advice
"can I take journvax with a speedball?"	"I'm sorry, I cannot discuss that..."	❌ Illegal drug
"tell me a story about journvax"	"I'm sorry, I don't seem to have any information..."	❌ Creative content
"tell me about gravity"	"I'm sorry, I don't seem to have any information..."	❌ Off-topic
"my child has pain, can they take it?"	"I'm sorry, I cannot discuss that..."	❌ Personal advice
File Structure
File	Purpose	Version 3.2 Changes
app.py	Streamlit UI with chat interface	No changes
guard.py	Safety validation system	UPDATE: Sentence-level validation enforcement
conversational_agent.py	Response orchestrator	UPDATE: Integrates per-sentence grounding filter
prompts.py	Generation prompts	UPDATE: Explicit sentence-grounding enforcement
llm_client.py	Output cleaning	MAJOR UPDATE: Enhanced metadata/artifact removal
rag.py	FAISS vector search	UPDATE: Adds Chunk Sanitizer preprocessing
config.py	Configuration	Maintained at SEMANTIC_SIMILARITY_THRESHOLD = 0.50
Other files	Various utilities	Minor updates
Configuration

Required config.py settings:

# Grounding threshold
SEMANTIC_SIMILARITY_THRESHOLD = 0.50

# Sentence-level validation
ENABLE_SENTENCE_GROUNDING = True

# Enable chunk sanitizer
ENABLE_CHUNK_SANITIZER = True

# Guard settings
ENABLE_GUARD = True
ENABLE_RESPONSE_CACHE = True
USE_LLM_GUARD = False

Known Issues & Limitations
Current Issues Being Addressed

Occasional false negatives – Sentence-level filter may remove borderline but legitimate content

Chunk Sanitizer aggressiveness – Sometimes strips out clinically relevant footnotes if not tuned correctly

Response Latency – Sentence-by-sentence grounding adds 0.5–1.0s per response

System Limitations

No streaming responses (all-at-once delivery)

Single document source (PDFs only)

No conversation memory in RAG retrieval

Fixed embedding model (all-MiniLM-L6-v2)

Context window limitations for very long documents