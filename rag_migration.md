# FAISS to Pinecone Migration Guide

## Overview
This guide covers migrating the Pharma Enterprise Assistant from FAISS (local vector storage) to Pinecone (cloud vector database).

## Benefits of Pinecone

### Scalability
- **Unlimited vectors**: No local storage constraints
- **Concurrent queries**: Handle multiple users simultaneously
- **Auto-scaling**: Automatically handles load spikes
- **Global distribution**: Low-latency access from anywhere

### Operational Benefits
- **Zero maintenance**: No index management or optimization needed
- **Built-in persistence**: No index corruption or rebuilding
- **Cloud backup**: Automatic data redundancy
- **Real-time updates**: Add/remove documents without rebuilding

### Performance
- **Optimized search**: Proprietary algorithms for fast retrieval
- **Semantic chunking**: Better context preservation with hybrid strategy
- **Metadata filtering**: Query by document source or section
- **Batch operations**: Efficient bulk uploads and queries

## Migration Steps

### 1. Set Up Pinecone Account
```bash
# Sign up at https://www.pinecone.io
# Get your API key from the dashboard
# Note your environment/region (e.g., us-east-1)
```

### 2. Configure Environment Variables
```bash
# Add to .env file or export
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_ENVIRONMENT="us-east-1"  # or your region
```

### 3. Install Updated Dependencies
```bash
# Remove old FAISS installation
pip uninstall faiss-cpu

# Install new requirements
pip install -r requirements.txt
```

### 4. Migrate Existing FAISS Index (Optional)
If you have an existing FAISS index with data:
```bash
# Automatic migration from FAISS to Pinecone
python build_index.py --migrate
```

### 5. Build Fresh Pinecone Index
For a clean build from PDFs:
```bash
# Build new index with semantic chunking
python build_index.py --path data/

# Or force rebuild if index exists
python build_index.py --rebuild
```

### 6. Verify Migration
```bash
# Check index status
python build_index.py --check-only
```

## Configuration Changes

### Updated config.py
- Added `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`
- Added `PINECONE_INDEX_NAME` (default: "pharma-assistant")
- Added `PINECONE_NAMESPACE` for document segregation
- Removed local FAISS index paths (kept for backward compatibility)

### Semantic Chunking Strategy
The system now uses enhanced semantic chunking by default:
- **Hybrid strategy**: Combines section-aware and paragraph chunking
- **Smart boundaries**: Preserves document structure
- **Metadata enrichment**: Tracks sections and sources
- **Optimal sizing**: 700-token chunks with 200-token overlap

## API Changes

### RAG System
The RAG interface remains the same:
```python
from rag import get_rag_system, retrieve_and_format_context

# Get RAG system (now Pinecone-based)
rag = get_rag_system()

# Retrieve context (unchanged API)
context = retrieve_and_format_context(query, k=5)
```

### Index Building
```python
from rag import build_index

# Build index (now uploads to Pinecone)
build_index(pdf_directory="data/", force_rebuild=False)
```

## Performance Considerations

### Network Latency
- Pinecone adds ~50-100ms network latency
- Mitigated by response caching
- Consider edge locations for global deployments

### Cost Optimization
- Free tier: 100K vectors, 1M queries/month
- Paid tiers scale with usage
- Use namespaces to segregate environments

### Best Practices
1. **Batch operations**: Upload vectors in batches of 100
2. **Metadata filtering**: Use for faster, targeted searches
3. **Index monitoring**: Check stats regularly
4. **Namespace strategy**: Separate dev/staging/prod

## Troubleshooting

### Common Issues

**API Key Error**
```
Error: PINECONE_API_KEY not configured
Solution: Set environment variable or add to .env file
```

**Network Timeout**
```
Error: Connection timeout to Pinecone
Solution: Check firewall/proxy settings, verify region
```

**Index Not Found**
```
Error: Index 'pharma-assistant' does not exist
Solution: Run build_index.py to create index
```

### Debug Commands
```bash
# Test Pinecone connection
python -c "from rag import get_rag_system; print(get_rag_system().get_index_stats())"

# Check index contents
python -c "from rag import get_rag_system; rag = get_rag_system(); print(f'Vectors: {rag.total_chunks}')"

# Test retrieval
python -c "from rag import retrieve_and_format_context; print(retrieve_and_format_context('test query')[:100])"
```

## Rollback Plan

If you need to rollback to FAISS:
1. Keep the old `faiss_index/` directory as backup
2. Restore the original `rag.py` from version control
3. Reinstall `faiss-cpu`: `pip install faiss-cpu>=1.7.4`
4. Update `config.py` to remove Pinecone settings

## Monitoring

### Key Metrics
- **Vector count**: Total documents indexed
- **Query latency**: Average retrieval time
- **Cache hit rate**: Percentage of cached responses
- **API usage**: Queries and storage consumption

### Health Checks
```python
# Add to monitoring script
from rag import get_rag_system

rag = get_rag_system()
stats = rag.get_index_stats()

assert stats["index_loaded"], "Index not loaded"
assert stats["total_chunks"] > 0, "No vectors in index"
assert stats["embedding_model"], "Embedding model not loaded"
```

## Support

- **Pinecone Documentation**: https://docs.pinecone.io
- **Pinecone Status**: https://status.pinecone.io
- **Community Forum**: https://community.pinecone.io

## Migration Timeline

1. **Hour 0-1**: Set up Pinecone account, configure API keys
2. **Hour 1-2**: Install dependencies, run migration script
3. **Hour 2-3**: Verify retrieval quality, test queries
4. **Hour 3-4**: Update monitoring, document changes
5. **Hour 4+**: Deploy to production, monitor performance