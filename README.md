# Safe Enterprise Assistant

A trust-based pharmaceutical RAG (Retrieval-Augmented Generation) system that empowers Llama 3.1 8B's natural conversational abilities while maintaining strict safety through intelligent post-generation validation.

## Philosophy

This system represents a paradigm shift from constraint-based to trust-based AI safety:
- **Trust the model's conversational abilities** - Let Llama 3.1 8B handle dialogue naturally
- **Simple, clear prompts** - Minimal instructions that don't constrain generation
- **Post-generation safety** - Intelligent guards that understand intent and context
- **Binary decisions** - Clear approve/reject for maximum safety

## System Architecture

The Safe Enterprise Assistant uses a streamlined architecture that maximizes model capabilities while ensuring pharmaceutical safety.

## Core Components

### Conversational Agent (`conversational_agent.py`)
- **Query Orchestration**: Routes queries through appropriate processing paths
- **Mode Detection**: Identifies greetings, questions, and conversation types
- **Session Management**: Monitors conversation limits and handles session transitions
- **RAG Integration**: Automatically retrieves context for non-greeting queries

### RAG Pipeline (`rag.py`)
- **Semantic Chunking**: Smart document processing that preserves meaning and context
- **FAISS Vector Database**: Fast similarity search for relevant content retrieval
- **Flexible Retrieval**: Returns relevant chunks without over-filtering
- **Graceful Degradation**: Continues working even if embedding model unavailable

### Semantic Chunker (`semantic_chunker.py`)
- **NLTK Integration**: Advanced NLP for sentence tokenization and entity extraction
- **Medical Document Parsing**: Specialized handling for FDA drug labels and pharma docs
- **Section Extraction**: Identifies headers and document structure
- **Recursive Splitting**: Maintains semantic boundaries while managing chunk sizes

### Conversational AI (`llm_client.py`)
- **Llama 3.1 8B Integration**: Via Hugging Face Inference Endpoints
- **Natural Parameters**: Temperature 0.7, top-p 0.9 for varied, human-like responses
- **Smart Token Management**: Prevents over-generation while allowing complete thoughts
- **Clean Response Handling**: Automatic cleanup of any formatting artifacts

### Enhanced Guard (`guard.py`)
- **Multi-Category Validation**: Checks for medical advice, competitor mentions, off-label use
- **Intent Recognition**: Understands whether response is answering, acknowledging gaps, or conversing
- **Context-Appropriate Validation**: Different standards for different types of responses
- **Binary Decisions**: Clear approve/reject for safety (no middle ground)
- **Smart Fallbacks**: Context-specific messages when responses are rejected

### Conversation Management (`conversation.py`)
- **Light State Tracking**: Just enough to maintain context
- **Natural Enhancement**: Helps with pronouns and references without constraining
- **Entity Extraction**: Tracks key pharmaceutical terms for better retrieval
- **Session Limits**: 10-turn default with automatic session management
- **Context Building**: Maintains conversation flow while respecting boundaries

### Context Formatter (`context_formatter.py`)
- **Document Formatting**: Clean presentation of retrieved context
- **Source Attribution**: Maintains document metadata and citations
- **Deduplication**: Removes redundant information from multiple sources
- **Structured Output**: Consistent format for LLM consumption

### Code Executor (`code_executor.py`)
- **Secure Execution**: Sandboxed Python code execution with security layers
- **AST Analysis**: Pre-execution code validation for safety
- **Resource Limits**: Memory, CPU, and execution time constraints
- **Import Restrictions**: Whitelist-based module control
- **Output Management**: Size limits and sanitization

### Streamlit Interface (`app.py`)
- **Clean UI**: Focused on conversation, not configuration
- **Smart Status Display**: Shows system health without clutter
- **Debug Mode**: Available when needed, hidden when not
- **Natural Error Messages**: Conversational even when things go wrong
- **Index Management**: Build and refresh document indexes from UI

## System Workflow

### Enhanced Flow

1. **User Query** → Natural language input
2. **Conversational Agent** → Analyzes query type and conversation mode
3. **RAG Retrieval** → Semantic search for relevant pharmaceutical documents
4. **Context Formatting** → Structures retrieved information for LLM
5. **Natural Generation** → Llama 3.1 8B generates response with context
6. **Guard Validation** → Multi-category safety and compliance checks
7. **User Response** → Approved response or safe fallback message

### The Trust Model

```
User Input → Agent Routing → RAG Search → LLM Generation → Guard Validation → Safe Output
     ↓              ↓             ↓              ↓                 ↓
  Session check   Mode detect  Semantic    Trust Llama 3.1   Multi-category
                              chunking         8B             safety checks
```

### Component Interaction Flow

```
app.py (UI)
    ↓
conversational_agent.py (Orchestration)
    ├→ conversation.py (Context tracking)
    ├→ rag.py (Document retrieval)
    │   └→ semantic_chunker.py (Document processing)
    ├→ context_formatter.py (Format context)
    ├→ llm_client.py (Generate response)
    └→ guard.py (Validate safety)
```

## Key Innovations

### Trust-Based Generation
- **Minimal System Prompt**: Just 2 sentences instead of pages of rules
- **Natural Context Format**: "Human:" / "Assistant:" instead of artificial structures
- **No Pre-Classification**: Let the model understand intent naturally
- **Freedom to Converse**: Greetings, clarifications, and transitions handled naturally

### Intent-Aware Safety
- **Conversational Responses**: Allowed without RAG grounding (e.g., "Hello!", "You're welcome!")
- **Gap Acknowledgments**: Can say "I don't have that information" naturally
- **Information Responses**: Must be grounded in retrieved documents
- **Context-Appropriate Standards**: Not every utterance needs a citation

### Simplified Architecture
- **Fewer Moving Parts**: Removed complex pre-processing and classification
- **Natural Data Flow**: Context → Model → Response → Validation
- **Clear Boundaries**: Generation is separate from safety validation
- **Maintainable Code**: Each component has a single, clear purpose

## Configuration

### Requirements
```bash
# .streamlit/secrets.toml or environment variables
HF_TOKEN = "your-hugging-face-token"
HF_INFERENCE_ENDPOINT = "your-model-endpoint-url"
```

### Key Parameters (config.py)
- **Generation**: Temperature 0.7, top-p 0.9, max 150 tokens
- **Retrieval**: Top 8 chunks, 600-800 token chunks with 100-150 overlap
- **Safety**: Binary approve/reject with multi-category validation
- **Sessions**: 10 turns max, 30-minute timeout
- **Code Execution**: 30s timeout, 512MB memory limit, restricted imports
- **Semantic Chunking**: NLTK-based sentence tokenization, entity extraction

## Installation & Usage

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### Usage
1. **Configure credentials**: Set HF_TOKEN and HF_INFERENCE_ENDPOINT in `.streamlit/secrets.toml`
2. **Add documents**: Place pharmaceutical PDFs in `data/` folder
3. **Start the app**: `streamlit run app.py`
4. **Build index**: Click "Build Index" in sidebar (first time only)
5. **Start chatting**: Natural conversation with pharmaceutical knowledge

### Testing
```bash
# Run all tests
python -m pytest test_*.py

# Specific test categories
python test_guard_violations.py  # Safety validation tests
python test_conversational_boundaries.py  # Conversation mode tests
python test_semantic_chunking.py  # Document processing tests
python test_rag.py  # RAG pipeline tests
```

## Safety & Compliance

The system maintains pharmaceutical safety through:
- **Multi-category validation** covering medical advice, competitors, off-label use, and promotional claims
- **Post-generation validation** instead of pre-generation rules
- **Intent understanding** to apply appropriate standards
- **Binary decisions** for clear safety boundaries
- **Natural fallbacks** that maintain conversational flow
- **Secure code execution** with sandboxing and resource limits
- **NLTK-based content analysis** for medical term recognition

## Why This Works

1. **Llama 3.1 8B is already good at conversation** - We don't need to teach it
2. **Post-generation checking is more flexible** - We can understand nuance
3. **Intent matters** - "Hello" doesn't need RAG grounding
4. **Simpler is safer** - Fewer components mean fewer failure points

## File Structure

```
safe_llama_assistant/
├── app.py                    # Streamlit UI and main application
├── conversational_agent.py   # Query orchestration and routing
├── rag.py                    # RAG pipeline and FAISS integration
├── semantic_chunker.py       # NLTK-based document processing
├── llm_client.py            # Hugging Face LLM integration
├── guard.py                 # Enhanced safety validation
├── conversation.py          # Conversation state management
├── context_formatter.py     # Context presentation layer
├── code_executor.py         # Secure code execution sandbox
├── config.py               # System configuration
├── prompt.py               # Prompt templates
├── data/                   # PDF documents folder
├── faiss_index/            # Vector database storage
└── test_*.py              # Comprehensive test suite
```

## The Result

A system that feels natural to use while maintaining strict pharmaceutical safety standards. Users get helpful, conversational responses grounded in your documentation, without the robotic feel of over-constrained AI.

## Performance Optimizations

- **Batch processing** for document indexing
- **Lazy loading** to avoid circular imports
- **FAISS optimization** for fast similarity search
- **Memory-efficient chunking** with configurable overlaps
- **Cached embeddings** to reduce API calls
- **Streamlined validation** with early exits