# llm_client.py
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import hashlib
from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """Client for interacting with Hugging Face Inference Endpoints."""
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        if not token:
            raise ValueError("HF_TOKEN is not configured.")
        if not endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured.")
        self.token = token
        self.endpoint = endpoint.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        logger.info(f"HuggingFaceClient initialized with endpoint: {self.endpoint[:50]}...")

    def _cache_key(self, prompt: str, parameters: Dict[str, Any]) -> str:
        return hashlib.md5(json.dumps({"prompt": prompt, "params": parameters}, sort_keys=True).encode()).hexdigest()

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        params = parameters.copy() if parameters else MODEL_PARAMS.copy()
        cache_key = self._cache_key(prompt, params)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = await self._generate_response_internal(prompt, params)
        # Limit cache size
        if len(self._cache) > 100:
            for k in list(self._cache.keys())[:20]:
                self._cache.pop(k, None)
        self._cache[cache_key] = result
        return result

    async def _generate_response_internal(self, prompt: str, parameters: Dict[str, Any]) -> str:
        payload = {"inputs": prompt, "parameters": parameters}
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)

        for attempt in range(3):
            try:
                logger.info(f"Sending request to HF endpoint (attempt {attempt + 1}/3)")
                async with aiohttp.ClientSession(connector=connector, headers=self.headers, timeout=timeout) as session:
                    async with session.post(self.endpoint, json=payload) as response:
                        response.raise_for_status()
                        data = await response.json()
                        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                            generated_text = data[0]["generated_text"]
                            if generated_text.startswith(prompt):
                                return generated_text[len(prompt):].strip()
                            return generated_text
                        # Some deployments return dicts
                        if isinstance(data, dict) and "generated_text" in data:
                            return data["generated_text"]
                        logger.warning(f"Unexpected response format: {data}")
                        return str(data)
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    logger.error("404 Not Found: Check your endpoint URL and deployment.", exc_info=True)
                    return "Error: The HuggingFace model endpoint is not accessible. Please check your endpoint configuration."
                logger.error(f"Request failed with status {e.status}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Request failed: {e}", exc_info=True)

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

        return "Error: Could not connect to the model service after multiple attempts. Please check your configuration."

# --- Convenience functions ---

_hf_client_instance = None
def get_hf_client():
    global _hf_client_instance
    if _hf_client_instance is None:
        _hf_client_instance = HuggingFaceClient()
    return _hf_client_instance

async def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
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
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 300,
        # Prefer 'stop' for broader compatibility with TGI
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###", "<|endoftext|>"]
    })
    return await call_huggingface(prompt, base_params)
