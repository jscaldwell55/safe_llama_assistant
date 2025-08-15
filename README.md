Safe Enterprise Assistant

A trust-based pharmaceutical RAG (Retrieval-Augmented Generation) assistant that uses a Hugging Face Inference Endpoint for generation, with strict post-generation safety and context grounding. Built with Streamlit.

What’s new (since the last README)

Welcome message on load: Users see a configurable WELCOME_MESSAGE immediately and can start typing (no “hello” back-and-forth).

Greeting ≠ question: Only standalone “hi/hello/hey” are treated as greetings. “hello, can you…” is handled as a real question with RAG.

Hardened Hugging Face client:

Accepts ≤ 4 stop tokens (avoids 422 errors) and tries stop → stop_sequences → none.

Fresh HTTP session per call; no “session closed” errors on retries.

Logs non-200 response bodies for easier debugging.

Safer guard:

If the reply contains facts, it’s treated as ANSWERING (validated + grounded), even if it starts with a greeting.

“Ungrounded” requires both low embedding similarity and low lexical overlap (reduces false rejects).

Graceful degradation if the embedding model isn’t available.

Clear, category-appropriate fallback messages.

Model error hygiene: If the model call fails, the app does not run the guard or write the error into chat history; users see a friendly error.

Endpoint rotation made easy: Update HF_INFERENCE_ENDPOINT only (secrets/env). Optional “Reload Model Client” button in the UI.

Renamed prompt module: prompt.py → prompts.py (prevents hot-reload KeyError).

Philosophy

Shift from constraint-heavy prompts to trust-based generation with strict post-checks:

Trust the model for natural conversation.

Minimal prompts, structured context.

Post-generation safety that understands intent and grounding.

Binary decisions (approve/reject) for compliance clarity.

System Architecture
User → app.py (UI)
           ↓
   conversational_agent.py (routing)
           ├─ conversation.py (state, welcome, entities)
           ├─ rag.py (retrieve) ── semantic_chunker.py (chunk)
           ├─ context_formatter.py (format retrieved context)
           ├─ llm_client.py (HF endpoint; robust params/retries)
           └─ guard.py (intent + grounding + safety → approve/reject)

Core Components
Streamlit App (app.py)

Welcome on load via conversation.py.

RAG-then-Check pipeline.

Model-error short-circuit: don’t guard or save when generation fails; show a friendly message.

Sidebar tools: Debug Mode, Show Context, New Conversation, Build/Refresh Index, Reload Model Client.

Conversational Agent (conversational_agent.py)

Standalone greeting detection (only short “hi/hello/hey” or “good morning/afternoon/evening” with no “?”).

Enhances queries with recent entities for better retrieval.

Always runs RAG retrieval for non-greetings.

Conversation Manager (conversation.py)

Seeds each session with WELCOME_MESSAGE.

Tracks last turns + lightweight entity extraction.

Session limits & timeout.

RAG Pipeline (rag.py)

SentenceTransformers embeddings (all-MiniLM-L6-v2).

FAISS flat index; persistent on disk.

Batch embedding generation for memory efficiency.

PDF ingestion via PyMuPDF.

One-call helper: retrieve_and_format_context(query) returns a concise, formatted context block.

Semantic Chunker (semantic_chunker.py)

Hybrid chunking (sections → paragraphs; or fallback).

NLTK sentence tokenizer; paragraph and recursive splitters.

FDA-style section detection.

Context Formatter (context_formatter.py)

Simple deduplication and compact, readable context assembly with separators and length limits.

LLM Client (llm_client.py)

Calls your Hugging Face Inference Endpoint.

Stop tokens ≤ 4; tries stop → stop_sequences → none (resolves endpoint differences).

Fresh per-call session; logs non-200 body text.

Small in-memory cache.

reset_hf_client() for hot endpoint rotation.

Enhanced Guard (guard.py)

Intent recognition: ANSWERING, ACKNOWLEDGING_GAP, CONVERSATIONAL_BRIDGE, OFFERING_ALTERNATIVES, CLARIFYING.

If facts exist → ANSWERING (mandatory grounding).

Dual grounding: embedding similarity (configurable threshold) and lexical overlap.

Medical safety checks (directives, overstatements).

Graceful degrade if embeddings unavailable.

Appropriate fallbacks (e.g., unsafe medical vs. no context).

Embedding Loader (embeddings.py)

Shared singleton loader for SentenceTransformer models.

Centralizes logging (you’ll see: “Loaded embedding model: all-MiniLM-L6-v2”).

Workflow

User input

Agent routing

Standalone greeting? → short friendly reply.

Otherwise enhance query and run RAG.

Context formatting → compact block.

Generation → HF Endpoint.

If model error → show friendly error; stop.

Guard

Detect intent; extract claims.

Grounding (semantic + lexical); medical checks.

Approve or return fallback.

Chat history updated only on approved replies.

Configuration
Secrets / Environment

Set only these two:

# .streamlit/secrets.toml  (recommended)
HF_TOKEN = "hf_xxx"
HF_INFERENCE_ENDPOINT = "https://<your-endpoint-id>.endpoints.huggingface.cloud"


or

# .env or platform variables
export HF_TOKEN=hf_xxx
export HF_INFERENCE_ENDPOINT="https://<your-endpoint-id>.endpoints.huggingface.cloud"


⚠️ We intentionally pin download hub envs inside code:
HF_ENDPOINT=https://huggingface.co (for model downloads) — do not use this for inference.

Quick Endpoint Rotation

Update HF_INFERENCE_ENDPOINT in secrets/env.

Restart the app or click “Reload Model Client” in the sidebar.

Key Parameters (config.py)

Model: temperature=0.7, top_p=0.9, repetition_penalty=1.1, max_new_tokens=300 (base call).

Stops: up to 4; client auto-fallbacks if your endpoint expects different keys.

RAG: top-k=8; chunk size ~800 chars; overlap ~150.

Embeddings: all-MiniLM-L6-v2; batch size=32.

Guard: SEMANTIC_SIMILARITY_THRESHOLD (default ~0.60–0.70) + lexical backup; binary approve/reject.

Conversation: 10 exchanges (20 turns), 30-min timeout.

UI: WELCOME_MESSAGE controls the initial banner text.

Installation
# Python deps
pip install -r requirements.txt

# NLTK data (first run can auto-download; these ensure consistency)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('words')"

Usage

Add your PDFs in data/.

Set HF_TOKEN and HF_INFERENCE_ENDPOINT in secrets or env.

Run:

streamlit run app.py


In the sidebar, Build / Refresh Index (first time, or after changing data/).

Start chatting in the input at the bottom.

Tips

Toggle Debug Mode to see guard details and model/guard errors.

Toggle Show Retrieved Context to inspect the exact grounding snippets.

Safety & Compliance

No ungrounded medical claims: answers must be traceable to retrieved context.

Rejects directive advice (e.g., “you should take 20 mg…”) unless quoting context.

Off-label + competitor mentions are treated conservatively.

If context is missing, the assistant uses a gap acknowledgment instead of guessing.

Troubleshooting

422 Unprocessable Entity from HF:

Some endpoints accept only 4 stop tokens and may prefer stop_sequences. The client auto-adjusts; check logs for the server’s error body.

Import hot-reload errors:

We renamed prompt.py → prompts.py. Ensure imports use prompts.

Model errors (“Error: …”):

The UI shows a friendly message; enable Debug Mode to see details.

No answers / frequent fallbacks:

Verify your data/ PDFs contain the topic; rebuild index; check that the embeddings loaded (look for “Loaded embedding model” in logs).

File Structure
safe_llama_assistant/
├── app.py                      # Streamlit UI + workflow orchestration
├── conversational_agent.py     # Greeting detection, query enhancement, RAG call
├── conversation.py             # State, entities, WELCOME_MESSAGE seeding
├── rag.py                      # FAISS index, retrieval, PDF ingestion
├── semantic_chunker.py         # Hybrid chunking (sections/paragraphs/sentences)
├── context_formatter.py        # Compact, deduplicated context assembly
├── llm_client.py               # HF Endpoint client (robust stops, retries, logging)
├── guard.py                    # Intent + grounding + medical safety, binary verdict
├── embeddings.py               # Singleton SentenceTransformer loader (shared)
├── config.py                   # App, model, guard, RAG, UI settings
├── prompts.py                  # System and guard prompts (renamed from prompt.py)
├── data/                       # Your PDFs
├── faiss_index/                # Persisted FAISS index + metadata
└── test_*.py                   # Tests (guard, chunking, rag, etc.)