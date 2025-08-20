Safe Enterprise Assistant (Pharma) — README

A production-ready pharmaceutical RAG assistant that answers questions only from approved documents (Medication Guide / PI) and enforces strict safety rules with a hybrid Guard (grounding + LLM + patterns). Built with Streamlit, FAISS, SentenceTransformers, and a Hugging Face Inference Endpoint.


What’s Included (Files)

app.py — Streamlit UI + request handling

conversational_agent.py — Orchestrator (retrieval → generation → guard → caching)

prompts.py — Bridge Synthesizer (generation) and guard prompts

rag.py — FAISS index, retrieval, PDF parsing (PyMuPDF), context assembly

semantic_chunker.py — Section/sentence/paragraph chunking with basic NLP signals

context_formatter.py — Dedupes/compacts retrieved context to fit prompt budget

embeddings.py — SentenceTransformer singleton (all-MiniLM-L6-v2 by default)

llm_client.py — Async HF endpoint client, retry, output cleanup, stop sequences

guard.py — HybridSafetyGuard (patterns + grounding + LLM), 9-category rules

conversation.py — Lightweight state/session management

config.py — All knobs: thresholds, model params, caching, UI text, logging

End-to-End Workflow
User Query
   │
   ▼
Persona Conductor (conversational_agent.py)
   ├─ Query Guard (guard.validate_query):
   │    • Pattern block (dose changes, pediatric, misuse, etc.)
   │    • Optional LLM query assessment (JSON-only, high threshold)
   │
   ├─ Decide retrieval:
   │    • If medical/product topic → retrieve context
   │
   ├─ RAG (rag.py):
   │    • FAISS top-K on normalized embeddings
   │    • semantic_chunker → context_formatter → compact context
   │
   ├─ Bridge Synthesizer (llm_client.py + prompts.py):
   │    • Generate grounded draft with ENHANCED_BRIDGE_PROMPT
   │
   ├─ Response Guard (guard.validate_response):
   │    • Pattern rules (see “Safety” below)
   │    • Grounding similarity (embeddings) + unsupported claims check
   │    • Optional LLM safety pass (never leaks editorial text)
   │    • Auto-corrections for specific cases (see below)
   │
   ├─ Cache approved response (LRU-ish FIFO)
   ▼
User Receives Final Answer (+ optional debug in UI)

Safety Capabilities (2 Pillars)
Pillar 1 — Mandatory Grounding

Response must be semantically similar to retrieved context.

Embedding model: all-MiniLM-L6-v2 (SentenceTransformers).

Threshold: SEMANTIC_SIMILARITY_THRESHOLD = 0.35 (lenient to allow paraphrase).

Unsupported claim detector: scans for specifics (numbers, “avoid X”, etc.) not present in context; repeated issues + low similarity ⇒ refusal.

Pillar 2 — Intelligent Guard

Hybrid checks applied to queries and responses:

Deterministic Patterns (block/redirect immediately):

Dosing questions (“how much should I take”, “double my dose”)

Pediatric/contraindicated use (“give to my baby”)

Unsafe admin (“crush/chew/smoke/inject/snort”, “share prescriptions”)

Improper tone (“don’t worry”, “perfectly safe”) / cross-brand analogies

LLM Guard (JSON-only, never surfaced to users):

Used for ambiguous cases; confidence threshold = 0.85 to reduce false positives.

Explicitly allows generic “contact your healthcare provider/pharmacist” statements (these are not individualized advice).

Auto-Corrections (not refusals) for two key categories:

Inadequate Risk Communication (2):
If side effects are listed without scope, auto-append:
“This is not a complete list. See the Medication Guide for full information.”

Mishandling Safety-Critical Info (7):
If red-flag terms (e.g., “trouble breathing”, “fainting”, “anaphylaxis”) appear without escalation, auto-append:
“If you have trouble breathing, swelling of the face, lips, tongue, or throat, fainting, chest pain, or signs of a severe allergic reaction, seek emergency medical care immediately.”

If an issue cannot be auto-corrected, the guard returns a neutral refusal (never editorial notes).

The 9 Enforced Categories (what triggers & what happens)

Inaccurate or Misleading Product Claims

Triggers: implied safety from silence; unsourced interactions/numbers.

Action: reject for poor grounding or unsubstantiated specifics.

Inadequate Risk Communication

Trigger: side-effect lists without scope guardrail.

Action: auto-append “not a complete list” disclaimer.

Off-Label or Unapproved Use

Trigger: pediatrics if adult-only, unlabeled indications, sharing.

Action: refuse + redirect to HCP; restate labeled population/indication only.

Improper Product Promotion (Tone/Scope)

Trigger: reassurance, lifestyle coaching, sexualized stories, speculative claims.

Action: refuse; enforce neutral, informational tone.

Cross-Product References / Misleading Brand Association

Trigger: “like [competitor] so it’s safe”, class-wide claims.

Action: refuse; speak only to the referenced product’s own label.

Practicing Medicine / Individualized Advice

Trigger: “you should double your dose”, treatment plans, triage.

Action: refuse; advise contacting HCP/pharmacist.

Mishandling Safety-Critical Information

Trigger: red-flag symptoms without escalation.

Action: auto-append emergency guidance; refuse only if cannot fix.

Failure to Address Misuse of Administration

Trigger: splitting/chewing if prohibited; sharing; giving to children.

Action: refuse + brief rationale + redirect to HCP; restate labeled constraints.

Unapproved Dosing/Admin Guidance

Trigger: any numeric/timing/food/alcohol/missed-dose details not verbatim from label, or tailored advice.

Action: refuse; point to Medication Guide/PI.

Golden Rule: If it isn’t verbatim in the PI/Medication Guide, don’t say it. Default to: refuse → brief rationale → redirect to HCP/pharmacist → cite Medication Guide (include “not a complete list”) → emergency language when indicated.

Configuration (key knobs)

See config.py:

Guard toggles

ENABLE_GUARD = True

USE_LLM_GUARD = True

SEMANTIC_SIMILARITY_THRESHOLD = 0.35

LLM_CONFIDENCE_THRESHOLD = 0.85

RAG

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_RETRIEVAL = 4

CHUNKING_STRATEGY = "hybrid"

MAX_CONTEXT_LENGTH = 3500

Models

HF_INFERENCE_ENDPOINT, HF_TOKEN

BRIDGE_SYNTHESIZER_PARAMS, GUARD_MODEL_PARAMS

Caching / Perf

ENABLE_RESPONSE_CACHE = True

MAX_CACHE_SIZE = 100

slow-request logging threshold, etc.

UI & Observability

Sidebar toggles: guard on/off, debug mode, cache flush, force rebuild index.

Debug panel per response: timing (RAG, generation, total), context length, whether context used, guard summary, grounding score, violation code.

Logs (stdout): RAG retrieval counts, index load/build, guard decisions, LLM latency.

Prompts (Generation)

ENHANCED_BRIDGE_PROMPT instructs the model to:

Use only documentation provided.

Avoid lists/headers; write natural sentences.

Include safety scope when discussing AEs.

Never invent dosing/admin details.

Extending / Customizing

Add products: drop PDFs into ./data/ and rebuild index.

Tune strictness: raise/lower SEMANTIC_SIMILARITY_THRESHOLD.

Adjust auto-fix scope: edit _append_disclaimer and emergency phrase in guard.py.

New red-flags: add to severe_signals in guard.py.

New patterns: extend pattern lists under each category.

Troubleshooting

“Won’t discuss side effects”
→ Ensure USE_LLM_GUARD=True but LLM_CONFIDENCE_THRESHOLD is high (0.85), and that the Medication Guide is in ./data/ with the index rebuilt. The guard will now auto-append the “not a complete list” line instead of refusing.

Grounding failures / “No info on that”
→ Confirm PDFs loaded, index built, and query references in-scope product terms.

PyMuPDF or FAISS install errors
→ Use faiss-cpu and pymupdf; on some systems you may need system libraries. Reinstall in a fresh venv.

Event loop / session issues
→ The HF client auto-recovers on loop errors; restart streamlit run app.py if needed.

Minimal API Surface (inside the app)

Index build: from rag import build_index; build_index(force_rebuild=True)

Retrieve: from rag import retrieve_and_format_context

Guard: from guard import evaluate_response (legacy adapter) or use enhanced_guard

Generate: from llm_client import call_bridge_synthesizer

Security Posture Summary

Hard grounding to approved docs, with similarity scoring and unsupported-claim checks.

Deterministic blocks for the highest-risk intents.

LLM Oversight for nuance (strict JSON, no editorial leakage).

Auto-fix where safe (scope + emergency guidance), refuse where not.

Single source of truth: Medication Guide/PI. Anything else → decline.