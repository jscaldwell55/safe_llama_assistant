import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
try:
    from functools import cache
except ImportError:
    cache = lru_cache(maxsize=None)
import hashlib
import time
from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """Client for interacting with Hugging Face Inference Endpoints"""
    
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        self.token = token
        self.endpoint = endpoint
        
        # Validate configuration
        if not self.token:
            raise ValueError("HF_TOKEN is not configured. Please set it in Streamlit secrets or environment variables.")
        
        if not self.endpoint:
            raise ValueError("HF_ENDPOINT is not configured. Please set it in Streamlit secrets or environment variables.")
        
        # Log the endpoint being used (without exposing sensitive parts)
        logger.info(f"Using endpoint: {self.endpoint[:50]}...")
        
        # Ensure the endpoint doesn't have a trailing slash for the API call
        if self.endpoint.endswith('/'):
            self.endpoint = self.endpoint.rstrip('/')
        
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self._session = None
        logger.info(f"HuggingFaceClient initialized with endpoint: {endpoint[:30]}...")
    
    async def _get_session(self):
        if self._session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
            self._session = aiohttp.ClientSession(connector=connector, headers=self.headers)
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
    
    def _cache_key(self, prompt: str, parameters: Dict[str, Any]) -> str:
        cache_data = json.dumps({"prompt": prompt, "params": parameters}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()

    async def _cached_generate(self, cache_key: str, prompt: str, params_json: str) -> str:
        # Simple in-memory cache for async methods
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        parameters = json.loads(params_json)
        result = await self._generate_response_internal(prompt, parameters)
        
        # Limit cache size
        if len(self._cache) > 100:
            # Remove oldest entries
            oldest_keys = list(self._cache.keys())[:20]
            for key in oldest_keys:
                del self._cache[key]
        
        self._cache[cache_key] = result
        return result

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        cache_key = self._cache_key(prompt, parameters)
        params_json = json.dumps(parameters, sort_keys=True)
        return await self._cached_generate(cache_key, prompt, params_json)

    async def _generate_response_internal(self, prompt: str, parameters: Dict[str, Any]) -> str:
        if not self.endpoint:
            return "Error: HuggingFace endpoint is not configured. Please check your settings."
        
        payload = {"inputs": prompt, "parameters": parameters}
        max_retries = 3
        session = await self._get_session()
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to HF endpoint (attempt {attempt + 1}/{max_retries})")
                logger.debug(f"Endpoint: {self.endpoint}")
                
                async with session.post(self.endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                        generated_text = result[0]["generated_text"]
                        # The response often includes the prompt, so we strip it.
                        if generated_text.startswith(prompt):
                            return generated_text[len(prompt):].strip()
                        return generated_text
                    else:
                        logger.warning(f"Unexpected response format: {result}")
                        return str(result)
                        
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    logger.error(f"404 Not Found: The endpoint URL '{self.endpoint}' is incorrect or the model is not deployed.")
                    return "Error: The HuggingFace model endpoint is not accessible. Please check your endpoint configuration."
                else:
                    logger.error(f"Request failed with status {e.status}: {e}", exc_info=True)
                    
            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
                
        return "Error: Could not connect to the model service after multiple attempts. Please check your configuration."

# --- Convenience Functions that use the lazy-loaded client ---

async def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to call the model via the lazy-loaded client."""
    try:
        client = get_hf_client()
        return await client.generate_response(prompt, parameters)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return f"Configuration error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in call_huggingface: {e}", exc_info=True)
        return "Error: An unexpected error occurred while calling the model."

async def call_base_assistant(prompt: str) -> str:
    """Calls the model with parameters optimized for conversational responses."""
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1,
        "max_new_tokens": 300, 
        # More specific stop sequences to prevent hallucination
        "stop_sequences": ["\nUser:", "\nHuman:", "\nAssistant:", "###", "<|endoftext|>"]
    })
    return await call_huggingface(prompt, base_params)

async def call_guard_agent(prompt: str) -> str:
    """Calls the model with parameters optimized for guard/safety evaluation."""
    guard_params = MODEL_PARAMS.copy()
    guard_params.update({
        "temperature": 0.3,  # Lower temperature for more deterministic evaluation
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 200,
        "stop_sequences": ["\n\n", "User:", "Assistant:"]
    })
    return await call_huggingface(prompt, guard_params)


# --- LAZY-LOADING FUNCTION ---
_hf_client_instance = None
def get_hf_client():
    """Lazy-loads and returns a single instance of the HuggingFaceClient."""
    global _hf_client_instance
    if _hf_client_instance is None:
        try:
            _hf_client_instance = HuggingFaceClient()
        except ValueError as e:
            logger.error(f"Failed to initialize HuggingFace client: {e}")
            raise
    return _hf_client_instance

# Cleanup function for proper shutdown
async def cleanup_hf_client():
    """Cleanup the HuggingFace client session."""
    global _hf_client_instance
    if _hf_client_instance:
        await _hf_client_instance.close()
        _hf_client_instance = None