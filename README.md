# 🛡️ Safe Enterprise Assistant

A production-ready RAG (Retrieval-Augmented Generation) system with LLM-based guard validation for enterprise use. The system combines document retrieval with dual-pass AI validation to ensure safe, grounded responses.

## 🏗️ Architecture

### Core Components

1. **📄 RAG Pipeline** (`rag.py`)
   - FAISS vector database for document storage
   - PDF chunking with overlapping windows
   - Semantic similarity search using sentence transformers

2. **🤖 Base Assistant** (`llm_client.py`)
   - Hugging Face Inference Endpoints integration
   - Meta-Llama-3-8B-Instruct model
   - Configurable generation parameters

3. **🛡️ Guard Agent** (`guard.py`)
   - Second LLM pass for response validation
   - JSON-structured evaluation output
   - Filters hallucinations, prompt leaks, and unsafe content

4. **🖥️ Streamlit Frontend** (`app.py`)
   - Clean, enterprise-focused UI
   - Debug mode with detailed inspection
   - Health monitoring and status checks

5. **⚙️ Configuration** (`config.py`)
   - Centralized settings management
   - Environment variable support
   - Flexible parameter tuning

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd safe_llama_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set your Hugging Face credentials:

```bash
export HF_TOKEN="your_huggingface_token"
export HF_ENDPOINT="your_inference_endpoint_url"
```

Or modify `config.py` directly.

### 3. Prepare Data

Place your PDF documents in the `data/` directory:

```bash
mkdir -p data
cp your_documents.pdf data/
```

### 4. Build Index

```bash
python -c "from rag import build_index; build_index()"
```

### 5. Launch Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```
safe_llama_assistant/
├── app.py              # Streamlit frontend
├── config.py           # Configuration settings
├── llm_client.py       # HuggingFace API client
├── rag.py              # RAG system implementation
├── guard.py            # Guard agent logic
├── prompt.py           # System prompts
├── ingest.py           # PDF processing utilities
├── test_rag.py         # System tests
├── requirements.txt    # Dependencies
├── data/               # PDF documents
├── faiss_index/        # Vector database files
└── README.md           # This file
```

## 🔧 Configuration Options

### Model Parameters (`config.py`)

```python
MODEL_PARAMS = {
    "max_new_tokens": 300,    # Response length
    "temperature": 0.3,       # Creativity vs consistency
    "top_p": 0.9,            # Nucleus sampling
    "do_sample": True        # Enable sampling
}
```

### RAG Settings

```python
CHUNK_SIZE = 500             # Text chunk size
CHUNK_OVERLAP = 50          # Overlap between chunks
TOP_K_RETRIEVAL = 5         # Number of chunks to retrieve
```

### Guard Configuration

```python
ENABLE_GUARD = True         # Enable/disable guard agent
GUARD_THRESHOLD = 0.7       # Similarity threshold
```

## 🛡️ Safety Features

### Base Assistant Safeguards

- **Context Grounding**: Only responds using provided context
- **Instruction Isolation**: Prevents prompt injection
- **Professional Tone**: Enterprise-appropriate responses
- **Fallback Handling**: Graceful degradation for missing information

### Guard Agent Validation

The guard agent evaluates responses for:

1. **Prompt Leakage**: System instructions or meta-commentary
2. **Hallucination**: Information not in the provided context
3. **Speculation**: Assumptions or unsubstantiated claims
4. **Inappropriate Advice**: Medical/legal advice beyond context
5. **Unsafe Content**: Harmful or offensive material
6. **Off-topic Responses**: Answers that don't address the question

### JSON Response Format

```json
{
  "verdict": "APPROVE",
  "reason": "Response grounded in context"
}
```

## 🧪 Testing

Run the test suite:

```bash
python test_rag.py
```

Test individual components:

```bash
# Test imports and basic functionality
python -c "import config, rag, guard, llm_client; print('✅ All modules loaded')"

# Test RAG retrieval
python -c "from rag import retrieve; print(retrieve('test query', k=3))"

# Test Streamlit app
python -c "import app; print('✅ App loads successfully')"
```

## 🔍 Usage Examples

### Basic Query

```python
from rag import retrieve
from prompt import format_base_prompt
from llm_client import call_base_assistant
from guard import evaluate_response

# 1. Retrieve context
chunks = retrieve("What are the side effects?")

# 2. Generate response
prompt = format_base_prompt("What are the side effects?", chunks)
response = call_base_assistant(prompt)

# 3. Validate with guard
approved, final_response, reasoning = evaluate_response(
    "\n\n".join(chunks), 
    "What are the side effects?", 
    response
)

print(f"Final response: {final_response}")
```

### Streamlit Interface Features

- **Main Query Interface**: Clean text input with enterprise styling
- **Debug Mode**: Detailed prompt inspection and model outputs
- **Context Display**: View retrieved document chunks
- **Health Monitoring**: System status and endpoint health checks
- **Safety Indicators**: Clear feedback when guard agent activates

## 🔄 Development Workflow

### Adding New Documents

1. Place PDF files in `data/` directory
2. Rebuild the index: `python -c "from rag import build_index; build_index(force_rebuild=True)"`
3. Test retrieval with sample queries

### Modifying Prompts

1. Edit prompts in `prompt.py`
2. Test with sample queries
3. Monitor guard agent behavior in debug mode

### Adjusting Guard Sensitivity

1. Modify `GUARD_THRESHOLD` in `config.py`
2. Test with edge cases
3. Evaluate false positive/negative rates

## 📊 Monitoring and Logs

The system provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Key log events:
- Document ingestion progress
- Retrieval performance metrics
- Guard agent decisions
- API call successes/failures

## 🚀 Production Deployment

### Environment Variables

```bash
HF_TOKEN=your_token
HF_ENDPOINT=your_endpoint
CHUNK_SIZE=500
ENABLE_GUARD=true
```

### Docker Deployment

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Performance Considerations

- **Index Size**: Monitor FAISS index memory usage
- **API Limits**: Configure request rate limiting
- **Caching**: Consider response caching for common queries
- **Scaling**: Use load balancers for multiple instances

## 🐛 Troubleshooting

### Common Issues

1. **No context retrieved**
   - Check if FAISS index exists
   - Verify PDF files are in `data/` directory
   - Rebuild index if needed

2. **HuggingFace API errors**
   - Verify token and endpoint URL
   - Check network connectivity
   - Monitor rate limits

3. **Guard agent always rejects**
   - Review guard prompts
   - Check temperature settings
   - Enable debug mode for inspection

### Debug Mode

Enable debug mode in the Streamlit sidebar to see:
- Full prompts sent to models
- Raw model responses
- Guard agent reasoning
- Retrieved context chunks

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

**Note**: This system is designed for enterprise use with safety-first principles. Always review responses in your specific domain context before deployment.