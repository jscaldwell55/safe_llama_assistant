Pharma Enterprise Assistant

A trust-based pharmaceutical RAG assistant that talks naturally while enforcing strict post-generation safety and context grounding.
Frontend and orchestration run in Streamlit; generation uses a Hugging Face Inference Endpoint.

What’s new

Welcome on load: Each new session shows a configurable greeting (WELCOME_MESSAGE) so users can start typing immediately.

Greeting ≠ question: Only standalone “hi/hello/hey/good morning/afternoon/evening” are treated as greetings. “hello, can you…” triggers full RAG.

Endpoint resilience:

Supports ≤ 4 stop tokens; auto-fallback from stop → stop_sequences → none to avoid 422s.

Fresh HTTP session per call (no “session closed” on retry).

Error body logging for easier debugging.

Reload Model Client button to hot-swap endpoints.

Guard hardened to philosophy:

If a reply contains facts, it’s treated as ANSWERING and must be grounded.

Dual grounding: semantic similarity (configurable threshold) and lexical overlap.

Immediate refusal for misuse/abusive routes (e.g., crush/snort/inject) and off-label probes without explicit “not indicated” context.

Graceful degrade if embeddings unavailable; category-appropriate fallback messages.

Model error hygiene: If generation fails, the app shows a friendly error and does not guard or save that turn.

Session resets on refresh: Conversation state is tied to Streamlit session; page refresh starts a new session with the default greeting.

Prompts module rename: prompt.py → prompts.py to avoid hot-reload import issues.

Philosophy

We trust the model to converse naturally and constrain with post-checks:

Minimal prompts, structured context.

No outside knowledge: respond only with content retrieved from the knowledge base; if nothing relevant is retrieved, redirect.

Binary safety: approve or reject—no partial credit.

Compliance by construction: enforce safety via guard rules, not sprawling prompts.

Critical Safety Requirements (enforced by guard)

No medical advice/diagnosis/treatment beyond documented information.

No dosage unless verbatim present in context (and accurately attributed).

No off-label or unapproved routes; if context says “not indicated,” it may be stated—then stop.

No competitor mentions unless present in context.

No promotional language (“best,” “guaranteed,” “breakthrough,” “most effective,” etc.).

Additionally: explicit misuse/abuse intent (crush/snort/inject, etc.) receives an immediate refusal with safety resources.

Architecture
User → app.py (UI)
           ↓
   conversational_agent.py (routing)
           ├─ conversation.py (session state, welcome, entities)
           ├─ rag.py (retrieve) ── semantic_chunker.py (chunk)
           ├─ context_formatter.py (format retrieved context)
           ├─ llm_client.py (HF endpoint; robust params/retries)
           └─ guard.py (intent + dual grounding + safety → approve/reject)

Core components
Streamlit App (app.py)

RAG-then-Check pipeline.

Short-circuit on model error (no guard, no history write).

Sidebar: Debug Mode, Show Context, New Conversation, Build/Refresh Index, Reload Model Client.

Header text: “💬 Ask me anything about Lexapro”.

Conversational Agent (conversational_agent.py)

Detects standalone greetings; everything else runs RAG.

Enhances follow-ups with recent entities for better retrieval.

Conversation Manager (conversation.py)

Seeds each session with WELCOME_MESSAGE (“Hello! How can I help you today?”).

Streamlit session-scoped → page refresh creates a fresh conversation.

Tracks recent turns and lightweight entity hints.

RAG Pipeline (rag.py)

Embeddings: SentenceTransformers (all-MiniLM-L6-v2).

Vector store: FAISS (flat), persisted to disk.

Batch embedding for memory efficiency.

PDF ingestion via PyMuPDF.

Helper: retrieve_and_format_context(query) returns a compact, citation-ready block.

Semantic Chunker (semantic_chunker.py)

Hybrid chunking (sections → paragraphs; fallback strategies).

NLTK sentence tokenizer; FDA-style section detection.

Context Formatter (context_formatter.py)

Simple dedup + separators; respects context length limits.

LLM Client (llm_client.py)

Hugging Face Inference Endpoint with:

≤ 4 stop tokens & auto-fallback strategy.

Fresh per-call session; response body logging.

Small in-memory cache.

reset_hf_client() for endpoint rotation.

Enhanced Guard (guard.py)

Intent detection for logging: ANSWERING, ACKNOWLEDGING_GAP, CONVERSATIONAL_BRIDGE, OFFERING_ALTERNATIVES, CLARIFYING.

Dual grounding gate for factual content (semantic + lexical).

Hard rejects for the 5 safety rules, plus misuse and off-label probes w/o “not indicated” context.

Graceful degrade if embeddings unavailable; category-specific fallbacks.

Embedding Loader (embeddings.py)

Shared singleton for SentenceTransformer model.

Centralized logging (e.g., “Loaded embedding model: all-MiniLM-L6-v2”).

Workflow

User input

Agent routing

Standalone greeting? → short, friendly reply.

Otherwise → enhance query and run RAG.

Context formatting → compact, deduplicated block.

Generation → Hugging Face endpoint.

On error → friendly message, stop.

Guard

Detect intent (for analysis).

Safety checks (5 rules + misuse/off-label).

Dual grounding of factual claims.

Approve or return fallback.

History updated only on approved replies.

Configuration

Secrets / env:

HF_TOKEN

HF_INFERENCE_ENDPOINT (the full Hugging Face endpoint URL)

Endpoint rotation:
Update HF_INFERENCE_ENDPOINT in secrets/env, then restart or use Reload Model Client in the sidebar.

Key parameters (see config.py):

Model: temperature 0.7, top_p 0.9, repetition_penalty 1.1, base max_new_tokens 300.

Stops: up to 4; client auto-fallback handles endpoint differences.

RAG: top-k 8; chunks ≈ 800 chars; overlap ≈ 150.

Embeddings: all-MiniLM-L6-v2; batch size 32.

Guard: SEMANTIC_SIMILARITY_THRESHOLD (≈0.60–0.70) + lexical backup; binary approve/reject.

Conversation: 10 exchanges (20 turns), 30-min timeout.

UI: WELCOME_MESSAGE controls the initial greeting.

Model download hub:
Internally pinned to https://huggingface.co for downloading open models (distinct from your inference endpoint).

Safety & compliance

Answers must be traceable to retrieved context; otherwise, return a gap message.

No directives or personalized medical guidance.

Off-label and abusive routes are refused outright (with appropriate safety messaging).

No promotional or competitor content unless explicitly present in context.

Troubleshooting (quick reference)

422 Unprocessable Entity: Some endpoints accept only four stop tokens or prefer stop_sequences. The client auto-adjusts and logs server error bodies.

Import/hot-reload issues: Use prompts.py (replaces prompt.py).

Frequent fallbacks: Confirm your PDFs contain the topic; rebuild the index; check logs for “Loaded embedding model…”.

File structure
safe_llama_assistant/
├── app.py                      # Streamlit UI + orchestration
├── conversational_agent.py     # Greeting detection, query enhancement, RAG call
├── conversation.py             # Session state, entities, WELCOME_MESSAGE
├── rag.py                      # FAISS index, retrieval, PDF ingestion
├── semantic_chunker.py         # Hybrid chunking
├── context_formatter.py        # Compact, deduplicated context assembly
├── llm_client.py               # HF endpoint client (stops, retries, logging)
├── guard.py                    # Intent + dual grounding + safety, binary verdict
├── embeddings.py               # Singleton SentenceTransformer loader
├── config.py                   # App/model/guard/RAG/UI settings
├── prompts.py                  # System and guard prompts
├── data/                       # PDFs
├── faiss_index/                # Persisted FAISS index + metadata
└── test_*.py                   # Tests (guard, chunking, rag, etc.)


Principle in one line:

Let the model speak naturally—only when the docs speak first.