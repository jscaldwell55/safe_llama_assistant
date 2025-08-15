# embeddings.py
import os
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_embedder: Optional[SentenceTransformer] = None

def get_embedding_model(name: Optional[str] = None) -> Optional[SentenceTransformer]:
    """
    Returns a singleton SentenceTransformer model instance.
    Uses the public Hugging Face hub for downloads (never your inference endpoint).
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    # Lazy import to avoid circulars
    from config import EMBEDDING_MODEL_NAME

    model_name = name or EMBEDDING_MODEL_NAME
    original_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        # Force downloads from the public hub
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        os.environ["HUGGINGFACE_HUB_URL"] = "https://huggingface.co"

        _embedder = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        _embedder = None
    finally:
        if original_endpoint is not None:
            os.environ["HF_ENDPOINT"] = original_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)

    return _embedder
