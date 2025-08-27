# Pharma Enterprise Assistant - Claude 3.5 Sonnet Edition

A document-grounded pharmaceutical information assistant powered by Claude 3.5 Sonnet, featuring RAG (Retrieval-Augmented Generation) with strict safety through system design.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Anthropic API key
- 100-200 pages of PDF documentation

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd pharma-assistant

Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Configure API key

bash# Create .streamlit directory
mkdir .streamlit

# Add your API key to secrets.toml
echo 'ANTHROPIC_API_KEY = "sk-ant-api03-YOUR-KEY"' > .streamlit/secrets.toml

Add your PDFs

bash# Create data directory and add PDFs
mkdir data
cp /path/to/your/*.pdf data/

Build the FAISS index

bashpython build_index.py

Run the application

bashstreamlit run app.py
📁 Project Structure
pharma-assistant/
├── app.py                     # Streamlit UI with chat interface
├── llm_client.py             # Claude 3.5 Sonnet integration
├── guard.py                  # Semantic grounding validator
├── conversational_agent.py  # Response orchestrator with caching
├── rag.py                    # FAISS-based retrieval system
├── config.py                 # Configuration settings
├── conversation.py           # Conversation state management
├── semantic_chunker.py       # Document chunking strategies
├── embeddings.py             # Embedding model management
├── context_formatter.py      # Context formatting utilities
├── prompts.py               # Legacy compatibility
├── build_index.py           # Index builder script
├── requirements.txt         # Python dependencies
├── data/                    # PDF documents (create this)
│   ├── document1.pdf
│   └── document2.pdf
├── faiss_index/             # Generated index files
│   ├── faiss.index
│   └── metadata.pkl
└── .streamlit/              # Streamlit configuration
    └── secrets.toml         # API keys (never commit!)
🔄 Workflow
Initial Setup (One-time)
mermaidgraph LR
    A[Add PDFs to data/] --> B[Run build_index.py]
    B --> C[Index created in faiss_index/]
    C --> D[Configure API key]
    D --> E[Run streamlit app]
Document Updates
When you need to update documents:

Add new PDFs to data/ folder
Rebuild index locally:
bashpython build_index.py --rebuild

Clear cache in the app UI
Test with relevant queries

Daily Usage

Start the app: streamlit run app.py
Ask questions about Journvax
Use "New Conversation" to reset context
Use "Clear Cache" for fresh responses

🏗️ Architecture
Safety Design (Simplified)
User Query
    ↓
RAG Retrieval (Top 5 chunks)
    ↓
Claude 3.5 Sonnet (System prompt enforced)
    ↓
Grounding Check (0.60 threshold)
    ↓
Response or Fallback
Key Components
ComponentPurposeKey FeaturesClaude IntegrationResponse generationStrong system prompt, conversation contextRAG SystemDocument retrievalFAISS flat index, semantic chunkingGrounding GuardSafety validationCosine similarity check (0.60 threshold)Response CachePerformanceLRU cache for 100 responsesConversation ManagerContext tracking20-turn history for Claude
📊 Performance Metrics
Expected Latencies

Cached Response: <50ms
RAG Retrieval: 100-300ms
Claude Generation: 1-2s
Grounding Check: 50-100ms
Total (uncached): 1.5-2.5s

Resource Usage

Memory: ~2GB (with model loaded)
Index Size: ~50-100MB (for 100-200 pages)
Cache Size: ~5MB (100 responses)

🚀 Deployment
Streamlit Cloud

Build index locally first:
bashpython build_index.py

Push to GitHub:
bashgit add .
git commit -m "Add application with pre-built index"
git push

Deploy on Streamlit Cloud:

Connect GitHub repository
Add ANTHROPIC_API_KEY to secrets
Deploy



Docker (Optional)
bash# Build image
docker build -t pharma-assistant .

# Run container
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/faiss_index:/app/faiss_index \
  pharma-assistant
🔧 Configuration
Key settings in config.py:
python# Model settings
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0.3  # Lower = more consistent

# RAG settings (optimized for 100-200 pages)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# Safety settings
SEMANTIC_SIMILARITY_THRESHOLD = 0.60
📈 Monitoring
Logs

Console: Real-time in terminal
File: app.log for persistent logging
Format: Timestamp, module, level, file:line, message

Key Log Patterns
[REQ_xxx]  - Request tracking
[PERF]     - Performance warnings
[STATS]    - System statistics
Debug Commands
python# Check index stats
from rag import get_index_stats
print(get_index_stats())

# Test retrieval
from rag import get_rag_system
rag = get_rag_system()
results = rag.retrieve("side effects")

# View cache stats
from conversational_agent import get_orchestrator
orch = get_orchestrator()
print(orch.get_stats())
🎯 Potential Improvements
Short-term (Easy)

Add query suggestions - Common questions as buttons
Export conversation - Save chat as PDF/text
Feedback system - Thumbs up/down for responses
Response streaming - Show Claude's response as it generates
Dark mode - UI theme toggle

Medium-term (Moderate)

Multi-document highlighting - Show which PDF each chunk comes from
Query expansion - Use synonyms for better retrieval
Reranking - Add a reranker for better chunk selection
Custom embeddings - Fine-tune embeddings on pharma domain
Analytics dashboard - Track usage patterns

Long-term (Complex)

Multi-modal support - Process images/tables from PDFs
Citation system - Link responses to specific PDF pages
A/B testing - Compare different prompts/thresholds
Active learning - Learn from user feedback
Multi-language - Support for non-English documents

Performance Optimizations

Async RAG - Parallel chunk retrieval
Redis cache - Distributed caching for scale
Vector DB - Replace FAISS with Pinecone/Weaviate for scale
Batch processing - Process multiple queries simultaneously
Edge caching - CDN for static assets

Safety Enhancements

Dual validation - Add factual accuracy check
Confidence scoring - Show confidence levels
Audit logging - Track all queries/responses
Content filtering - Additional PII/PHI detection
Fallback models - Backup if Claude is unavailable

🐛 Troubleshooting
Common Issues
IssueSolution"No index found"Run python build_index.py"API key not configured"Add to .streamlit/secrets.tomlSlow responsesCheck internet connection, API statusPoor retrievalAdjust chunk size, rebuild indexMemory errorsReduce batch size, use smaller embedding model
Health Checks
bash# Test system components
python test_system.py

# Check index only
python build_index.py --check-only

# Rebuild if corrupted
python build_index.py --rebuild