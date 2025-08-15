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
        if not hasattr(self, "_cache"):
            self._cache = {}
        ck = self._cache_key(prompt, params)
        if ck in self._cache:
            return self._cache[ck]

        result = await self._generate_response_internal(prompt, params)

        # Limit cache size
        if len(self._cache) > 100:
            for k in list(self._cache.keys())[:20]:
                self._cache.pop(k, None)
        self._cache[ck] = result
        return result

    async def _post_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One HTTP POST with a fresh session. Returns (status, data_text, parsed_json_if_any)."""
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(headers=self.headers, timeout=timeout) as session:
            async with session.post(self.endpoint, json=payload) as response:
                text = await response.text()
                try:
                    data = json.loads(text)
                except Exception:
                    data = None
                return {"status": response.status, "text": text, "json": data}

    async def _generate_response_internal(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """
        Try multiple parameter variants to handle different Inference Endpoint backends:
        1) TGI-style:          parameters with 'stop'
        2) transformers-style: parameters with 'stop_sequences'
        3) minimal:            parameters without stop keys
        """
        # Build variants
        base = parameters.copy()
        stop_list = base.pop("stop", None)

        variants: list[Dict[str, Any]] = []
        # 1) TGI-style (stop)
        if stop_list:
            v1 = base.copy()
            v1["stop"] = stop_list
            variants.append(v1)
        else:
            variants.append(base.copy())

        # 2) stop_sequences
        if stop_list:
            v2 = base.copy()
            v2["stop_sequences"] = stop_list
            variants.append(v2)

        # 3) minimal (no stop)
        variants.append(base.copy())

        errors = []
        for i, params in enumerate(variants, start=1):
            payload = {"inputs": prompt, "parameters": params}
            logger.info(f"Sending request to HF endpoint (variant {i}/{len(variants)})")
            try:
                result = await self._post_once(payload)
                status, text, data = result["status"], result["text"], result["json"]

                if status == 200:
                    # TGI often returns [{"generated_text": "..."}]
                    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                        generated_text = data[0]["generated_text"]
                        return generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
                    # Some backends return {"generated_text": "..."}
                    if isinstance(data, dict) and "generated_text" in data:
                        return data["generated_text"]
                    # Fallback: best effort
                    logger.warning(f"200 OK but unrecognized response format: {text[:300]}...")
                    return text

                # Non-200 => capture and try next variant (422 likely means bad param name)
                logger.error(f"HF endpoint returned {status}. Body: {text[:800]}")
                errors.append(f"{status}: {text[:300]}")

            except Exception as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                errors.append(repr(e))

            # small backoff between variants
            await asyncio.sleep(0.3)

        # None of the variants worked
        return "Error: Could not generate from the model (tried multiple parameter formats). " + (errors[0] if errors else "")

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
        # prefer 'stop', but _generate_response_internal will try fallbacks if needed
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###", "<|endoftext|>"]
    })
    return await call_huggingface(prompt, base_params)

def reset_hf_client():
    """Optional: allows reloading the client after rotating the endpoint via the UI."""
    global _hf_client_instance
    _hf_client_instance = None
