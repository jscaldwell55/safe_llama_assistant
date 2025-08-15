# llm_client.py
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import hashlib
from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        if not token:
            raise ValueError("HF_TOKEN is not configured.")
        if not endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured.")
        self.token = token
        self.endpoint = endpoint.rstrip('/')
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
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
        if len(self._cache) > 100:
            for k in list(self._cache.keys())[:20]:
                self._cache.pop(k, None)
        self._cache[ck] = result
        return result

    async def _post_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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
        base = parameters.copy()

        # Ensure at most 4 stop tokens (some endpoints hard-limit to 4)
        stop_list = base.pop("stop", None)
        if stop_list:
            stop_list = list(stop_list)[:4]

        variants: list[Dict[str, Any]] = []
        if stop_list:
            v1 = base.copy(); v1["stop"] = stop_list; variants.append(v1)
            v2 = base.copy(); v2["stop_sequences"] = stop_list; variants.append(v2)
        variants.append(base.copy())  # minimal (no stop)

        errors = []
        for i, params in enumerate(variants, start=1):
            payload = {"inputs": prompt, "parameters": params}
            logger.info(f"Sending request to HF endpoint (variant {i}/{len(variants)})")
            try:
                result = await self._post_once(payload)
                status, text, data = result["status"], result["text"], result["json"]

                if status == 200:
                    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                        generated_text = data[0]["generated_text"]
                        return generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
                    if isinstance(data, dict) and "generated_text" in data:
                        return data["generated_text"]
                    logger.warning(f"200 OK but unrecognized response format: {text[:300]}...")
                    return text

                logger.error(f"HF endpoint returned {status}. Body: {text[:800]}")
                errors.append(f"{status}: {text[:300]}")
            except Exception as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                errors.append(repr(e))
            await asyncio.sleep(0.3)

        return "Error: Could not generate from the model (tried multiple parameter formats). " + (errors[0] if errors else "")

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
        # keep to <= 4
        "stop": ["\nHuman:", "\nUser:", "\nAssistant:", "###"]
    })
    return await call_huggingface(prompt, base_params)

def reset_hf_client():
    global _hf_client_instance
    _hf_client_instance = None
