# Safe Enterprise Assistant

A production-ready RAG (Retrieval-Augmented Generation) system with LLM-based guard validation for enterprise use. The system combines document retrieval with dual-pass AI validation to ensure safe, grounded responses.

## System Architecture

The Safe Enterprise Assistant is a multi-layered system designed to provide secure, context-aware responses to user queries through document retrieval and AI validation.

## Core Components

### RAG Pipeline (`rag.py`)
- **FAISS Vector Database**: Stores document embeddings for fast similarity search
- **PDF Processing**: Chunks documents with overlapping windows for better context preservation
- **Semantic Search**: Uses sentence transformers to find relevant document sections
- **Context Retrieval**: Returns top-k most relevant chunks for query answering

### Base Assistant (`llm_client.py`)
- **Hugging Face Integration**: Connects to Meta-Llama-3-8B-Instruct via Inference Endpoints
- **Response Generation**: Produces contextual answers based on retrieved documents
- **Parameter Control**: Configurable temperature, token limits, and sampling methods
- **Context Grounding**: Restricts responses to information found in provided documents

### Guard Agent (`guard.py`)
- **Response Validation**: Second LLM pass to evaluate response quality and safety
- **Safety Filtering**: Detects hallucinations, prompt leaks, and unsafe content
- **JSON Evaluation**: Structured output with approval verdict and reasoning
- **Multi-criteria Assessment**: Checks for speculation, off-topic responses, and inappropriate advice

### Streamlit Frontend (`app.py`)
- **User Interface**: Clean, enterprise-focused web application
- **Debug Mode**: Detailed inspection of prompts, responses, and guard decisions
- **Health Monitoring**: System status checks and endpoint connectivity
- **Interactive Features**: Query input, context display, and safety indicators

### Configuration Management (`config.py`)
- **Centralized Settings**: Model parameters, RAG settings, and guard thresholds
- **Environment Variables**: Secure credential management
- **Flexible Tuning**: Adjustable parameters for different use cases

## System Workflow

### Query Processing Flow

1. **Document Ingestion**
   - PDF documents are processed and chunked into overlapping segments
   - Text embeddings are generated using sentence transformers
   - FAISS index is built for efficient similarity search

2. **Query Retrieval**
   - User query is embedded using the same transformer model
   - Semantic similarity search retrieves top-k relevant document chunks
   - Retrieved context is ranked by relevance score

3. **Response Generation**
   - Base assistant receives user query and retrieved context
   - LLM generates response grounded in provided documents
   - Response is constrained to information available in context

4. **Guard Validation**
   - Guard agent evaluates the generated response for safety and accuracy
   - Checks for hallucinations, prompt leaks, speculation, and inappropriate content
   - Returns structured verdict with approval status and reasoning

5. **Final Output**
   - Approved responses are presented to the user
   - Rejected responses trigger fallback messages
   - Debug information is available for system monitoring

### Safety Mechanisms

- **Context Grounding**: Responses are limited to information in retrieved documents
- **Instruction Isolation**: System prompts prevent prompt injection attacks
- **Dual Validation**: Two-stage AI review ensures response quality
- **Structured Evaluation**: JSON-formatted guard decisions enable consistent filtering
- **Fallback Handling**: Graceful degradation when context is insufficient

### Data Flow

```
PDF Documents → Chunking → Embedding → FAISS Index
     ↓
User Query → Embedding → Similarity Search → Context Retrieval
     ↓
Query + Context → Base Assistant → Initial Response
     ↓
Context + Query + Response → Guard Agent → Validation
     ↓
Approved Response → User Interface → Final Output
```