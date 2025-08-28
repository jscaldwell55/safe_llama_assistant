# embeddings.py - Fixed for Apple Silicon/MPS issues
import os
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)
_embedder: Optional[SentenceTransformer] = None

def get_embedding_model(name: Optional[str] = None) -> Optional[SentenceTransformer]:
    """
    Returns a singleton SentenceTransformer model instance.
    Forces CPU usage to avoid MPS segmentation faults.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    # Import config
    from config import EMBEDDING_MODEL_NAME
    model_name = name or EMBEDDING_MODEL_NAME
    
    # CRITICAL: Force CPU usage to avoid MPS issues
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Remove ALL HuggingFace environment variables
    env_vars_to_remove = [
        'HF_TOKEN', 'HUGGING_FACE_TOKEN', 'HUGGINGFACE_TOKEN',
        'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN',
        'HF_API_TOKEN', 'HF_HOME', 'HF_HUB_CACHE',
        'HF_ENDPOINT', 'HUGGINGFACE_HUB_URL', 'TRANSFORMERS_CACHE',
    ]
    
    for var in env_vars_to_remove:
        os.environ.pop(var, None)
    
    try:
        logger.info(f"Loading embedding model: {model_name} (CPU-only)")
        
        # Force CPU device to avoid MPS issues
        device = 'cpu'
        
        # Load model with explicit CPU device
        _embedder = SentenceTransformer(
            model_name,
            device=device
        )
        
        # Double-check it's on CPU
        _embedder = _embedder.to(device)
        
        logger.info(f"Successfully loaded embedding model: {model_name} on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        _embedder = None
    
    return _embedder