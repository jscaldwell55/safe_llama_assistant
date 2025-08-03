# Safe Enterprise Assistant

A production-ready pharmaceutical RAG (Retrieval-Augmented Generation) system with advanced LLM-based guard validation for enterprise use. The system combines document retrieval with multi-layer AI validation, semantic grounding verification, and pharmaceutical compliance to ensure safe, accurate, and contextually grounded responses.

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
- **Pharmaceutical Compliance**: Enforces strict medical safety guidelines
  - No medical advice, diagnosis, or treatment recommendations
  - No dosage instructions beyond provided context
  - No off-label use discussions or competitor mentions
  - Only FDA-approved information from context
- **Semantic Grounding Validation**: Uses all-MiniLM-L6-v2 embeddings for similarity scoring
  - Extracts and validates factual claims against context
  - Calculates semantic similarity scores (threshold: 0.7)
  - Tracks ungrounded statements for detailed feedback
- **Medical Claim Validation**: Context-aware medical claim verification
  - Extracts 7 types of medical claims (efficacy, side effects, dosage, etc.)
  - Validates each claim against provided RAG content
  - Distinguishes between quoting documentation vs. making recommendations
- **Multi-layer Safety**: Progressive validation pipeline
  - Pharmaceutical compliance violations
  - Medical claim context validation
  - Semantic RAG grounding
  - Final LLM-based guard evaluation

### Streamlit Frontend (`app.py`)
- **User Interface**: Clean, enterprise-focused web application
- **Debug Mode**: Detailed inspection of prompts, responses, and guard decisions
- **Health Monitoring**: System status checks and endpoint connectivity
- **Interactive Features**: Query input, context display, and safety indicators

### Conversation Management (`conversation.py`)
- **Session Management**: Tracks conversation state and context
  - Configurable turn limits (default: 999999 for testing)
  - Session timeout handling
  - Topic and entity tracking across turns
- **Enhanced Follow-up Detection**: Context-aware classification
  - Identifies follow-up questions while protecting critical safety queries
  - Excludes questions with safety keywords (children, pregnancy, etc.) from follow-up classification
  - Maintains conversation continuity without compromising safety
- **Query Enhancement**: Improves retrieval for follow-up questions
  - Adds conversation context to enhance RAG retrieval
  - Tracks active entities and current topics

### Conversational Agent (`conversational_agent.py`)
- **Natural Conversation Flow**: Maintains engagement while respecting RAG constraints
  - Conversational bridging for better user experience
  - Intelligent topic acknowledgment and redirection
  - Contextual fallback messages
- **Response Enhancement**: Adds natural conversational elements
  - Mode-specific conversation starters
  - Appropriate follow-up suggestions
  - Professional tone maintenance

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

4. **Multi-Layer Guard Validation**
   - **Pharmaceutical Compliance Check**: Detects medical advice, dosage recommendations, off-label uses
   - **Medical Claim Validation**: Verifies all medical claims against provided context
   - **Semantic Grounding**: Calculates embedding similarity between response and context (0.7+ required)
   - **Final Guard Evaluation**: LLM-based assessment for overall safety and appropriateness
   - Returns structured verdict with approval status and detailed reasoning

5. **Final Output**
   - Approved responses are presented to the user
   - Rejected responses trigger fallback messages
   - Debug information is available for system monitoring

### Safety Mechanisms

- **RAG-Grounded Policy**: All factual claims must be grounded in provided context
  - Allows language understanding for explanation and organization
  - Permits meta-knowledge for terminology and navigation
  - Enables conversational bridging while maintaining accuracy
- **Pharmaceutical Compliance**: Strict enforcement of medical safety guidelines
  - Context-aware pattern matching distinguishes quotes from recommendations
  - Critical safety keywords prevent misclassification of important queries
  - Multi-category violation detection (medical advice, competitors, off-label, etc.)
- **Semantic Validation**: Embedding-based grounding verification
  - Extracts factual claims vs. conversational elements
  - Calculates cosine similarity with context sentences
  - Provides confidence scores and identifies ungrounded statements
- **Progressive Validation Pipeline**: Multiple layers of safety checks
  - Pattern-based compliance filtering
  - Context-aware medical claim validation
  - Semantic similarity grounding
  - LLM-based final evaluation
- **Intelligent Fallbacks**: Context-aware response handling
  - Conversational bridging for partial matches
  - Suggestions for related available topics
  - Professional redirection when no information available

### Data Flow

```
PDF Documents → Chunking → Embedding (all-MiniLM-L6-v2) → FAISS Index
     ↓
User Query → Conversation Analysis → Query Enhancement → Embedding
     ↓
Enhanced Query → Similarity Search → Context Retrieval
     ↓
Query + Context + Conversation History → Base Assistant → Initial Response
     ↓
Response → Pharmaceutical Compliance Check → Pass/Reject
     ↓
Response → Medical Claim Extraction → Context Validation → Pass/Reject
     ↓
Response → Semantic Grounding (Embeddings) → Similarity Score → Pass/Reject
     ↓
Response → LLM Guard Evaluation → Final Verdict → Pass/Reject
     ↓
Approved Response → Conversational Enhancement → User Interface → Final Output
```

## Key Features & Improvements

### Enhanced Conversational Experience
- **Natural Language Flow**: Maintains conversational engagement while strictly adhering to RAG content
- **Intelligent Bridging**: Acknowledges topics and redirects to available information gracefully
- **Context-Aware Responses**: Different handling for greetings, follow-ups, and information requests

### Advanced Safety Validation
- **Multi-Layer Validation**: Progressive checks ensure response safety and accuracy
- **Semantic Grounding**: Uses embedding similarity to verify factual claims against context
- **Context-Aware Patterns**: Distinguishes between quoting documentation and making recommendations
- **Medical Safety Focus**: Specialized validation for pharmaceutical and healthcare content

### Flexible Configuration
- **Adjustable Thresholds**: Semantic similarity threshold (default: 0.7) can be tuned
- **Turn Limits**: Configurable conversation limits for testing and production
- **Modular Design**: Easy to extend or modify individual validation layers

### Production-Ready Features
- **Comprehensive Logging**: Detailed tracking of validation decisions and scores
- **Graceful Degradation**: Intelligent fallbacks when context is insufficient
- **Debug Mode**: Full visibility into the validation pipeline for troubleshooting