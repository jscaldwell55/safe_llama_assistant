# llm_client.py
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
try:
    from functools import cache
except ImportError:
    cache = lru_cache(maxsize=None)
import hashlib
from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logger = logging.getLogger(__name__)

# ---------- NEW: output sanitizer ----------
import re

_META_LINE_PATTERNS = [
    r'^\s*note\s*:.*',                      # Note: ...
    r'^\s*please ignore.*',                 # Please ignore...
    r'^\s*here (?:is|\'s) the revised response.*',
    r'^\s*additional queries and responses.*',
    r'^\s*(user|assistant)\s*:',            # User: / Assistant:
    r'^\s*###\s*end response.*',
]

def clean_model_output(text: str) -> str:
    if not text:
        return text or ""

    # Remove prompt echo
    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    # Cut off at common out-of-band markers
    cut_markers = [
        "### End Response", "### End", "Additional Queries and Responses",
        "\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"
    ]
    for m in cut_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx].rstrip()
            break

    # Remove parenthetical labels e.g. (Label: Retrieved information)
    text = re.sub(r"\(\s*(label|source)\s*:[^)]+\)", "", text, flags=re.IGNORECASE)

    # Drop meta lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    cleaned_lines = []
    for ln in lines:
        lower = ln.lower()
        if any(re.match(pat, lower) for pat in _META_LINE_PATTERNS):
            continue
        cleaned_lines.append(ln)
    text = "\n".join(cleaned_lines).strip()

    # Deduplicate obvious repeated paragraph blocks
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    seen = set()
    unique_paras = []
    for p in paragraphs:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique_paras.append(p)
    text = "\n\n".join(unique_paras).strip()

    # Fix dangling trailing "(" fragments
    if text.endswith("("):
        text = text[:-1].rstrip()
    if text.count("(") > text.count(")"):
        # remove from last unmatched "(" to end
        last = text.rfind("(")
        if last != -1:
            text = text[:last].rstrip()

    return text

# ---------- HF client ----------
class HuggingFaceClient:
    """Client for interacting with Hugging Face Inference Endpoints"""

    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        self.token = token
        self.endpoint = endpoint

        if not self.token:
            raise ValueError("HF_TOKEN is not configured. Please set it in Streamlit secrets or environment variables.")
        if not self.endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured. Please set it in Streamlit secrets or environment variables.")

        logger.info(f"HuggingFaceClient initialized with endpoint: {self.endpoint[:60]}...")
        if self.endpoint.endswith('/'):
            self.endpoint = self.endpoint.rstrip('/')

        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._session = None

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
        return hashlib.md5(json.dumps({"prompt": prompt, "params": parameters}, sort_keys=True).encode()).hexdigest()

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        cache_key = self._cache_key(prompt, parameters)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if cache_key in getattr(self, "_cache", {}):
            return self._cache[cache_key]
        result = await self._generate_response_internal(prompt, parameters)
        # sanitize before caching/returning
        result_clean = clean_model_output(result)
        # small cache with eviction
        if len(self._cache) > 100:
            for k in list(self._cache.keys())[:20]:
                del self._cache[k]
        self._cache[cache_key] = result_clean
        return result_clean

    async def _generate_response_internal(self, prompt: str, parameters: Dict[str, Any]) -> str:
        payload_variants = []

        # Variant 1: up to 4 stop tokens
        v1 = parameters.copy()
        v1["stop"] = ["\nUser:", "\nHuman:", "\nAssistant:", "###"]
        if "stop_sequences" in v1:
            v1.pop("stop_sequences", None)
        payload_variants.append({"inputs": prompt, "parameters": v1})

        # Variant 2: stop_sequences instead
        v2 = parameters.copy()
        v2.pop("stop", None)
        v2["stop_sequences"] = ["\nUser:", "\nHuman:", "\nAssistant:", "###"]
        payload_variants.append({"inputs": prompt, "parameters": v2})

        # Variant 3: no stops
        v3 = parameters.copy()
        v3.pop("stop", None)
        v3.pop("stop_sequences", None)
        payload_variants.append({"inputs": prompt, "parameters": v3})

        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=60)

        for i, payload in enumerate(payload_variants, start=1):
            try:
                logger.info(f"Sending request to HF endpoint (variant {i}/3)")
                async with session.post(self.endpoint, json=payload, timeout=timeout) as response:
                    body = await response.text()
                    if response.status != 200:
                        logger.error(f"HF endpoint returned {response.status}. Body: {body}")
                        continue
                    try:
                        result = json.loads(body)
                    except Exception:
                        logger.error(f"Unexpected non-JSON body: {body[:300]}")
                        return body
                    if isinstance(result, list) and result and "generated_text" in result[0]:
                        gen = result[0]["generated_text"]
                        if gen.startswith(prompt):
                            gen = gen[len(prompt):].lstrip()
                        return gen
                    else:
                        logger.warning(f"Unexpected response format: {result}")
                        return str(result)
            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)

        return "Error: Could not connect to the model service after multiple attempts. Please check your configuration."

# --- Convenience functions using the lazy-loaded client ---
_hf_client_instance = None

def get_hf_client():
    global _hf_client_instance
    if _hf_client_instance is None:
        _hf_client_instance = HuggingFaceClient()
    return _hf_client_instance

def reset_hf_client():
    global _hf_client_instance
    if _hf_client_instance is not None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        async def _close():
            try:
                await _hf_client_instance.close()
            except Exception:
                pass
        if loop and loop.is_running():
            asyncio.create_task(_close())
        else:
            asyncio.run(_close())
    _hf_client_instance = None

async def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    try:
        client = get_hf_client()
        raw = await client.generate_response(prompt, parameters)
        return raw  # already cleaned
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return f"Configuration error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in call_huggingface: {e}", exc_info=True)
        return "Error: An unexpected error occurred while calling the model."

async def call_base_assistant(prompt: str) -> str:
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1,
        "max_new_tokens": 300
        # stop/stop_sequences handled by variants above
    })
    return await call_huggingface(prompt, base_params)

async def call_guard_agent(prompt: str) -> str:
    guard_params = MODEL_PARAMS.copy()
    guard_params.update({
        "temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.0,
        "max_new_tokens": 200
    })
    return await call_huggingface(prompt, guard_params)
