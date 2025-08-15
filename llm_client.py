import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional, Tuple
import hashlib
import re
import time

from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS, BASE_MAX_NEW_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Output cleaning
# -------------------------------
_META_LINE_PATTERNS = [
    r"^\s*note\s*:\s*",
    r"^\s*this (answer|response) (?:is|was)\b",
    r"^\s*i (?:am|’m|\'m)\s+(?:an|a)\s+ai\b",
    r"^\s*as (?:an|a) (?:ai|language model)\b",
    r"^\s*please (?:ignore|disregard)\b",
    r"^\s*for (?:educational|informational) purposes\b",
    r"^\s*this (?:message|content)\s+(?:complies|adheres|follows)\b",
    r"^\s*i cannot (?:do that|comply)\b",
    r"^\s*debug\s*:",
]

def clean_model_output(text: str) -> str:
    if not text:
        return text or ""

    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    for m in ["### End Response", "### End", "Additional Queries and Responses",
              "\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"]:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx].rstrip()
            break

    text = re.sub(r"\(\s*(label|source)\s*:[^)]+\)", "", text, flags=re.IGNORECASE)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    cleaned_lines = []
    for ln in lines:
        lower = ln.lower()
        if any(re.match(pat, lower) for pat in _META_LINE_PATTERNS):
            continue
        if "this response" in lower and any(k in lower for k in ["neutral", "complies", "adheres", "follows", "meets"]):
            continue
        cleaned_lines.append(ln)
    text = "\n".join(cleaned_lines).strip()

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    seen = set()
    unique_paras = []
    for p in paragraphs:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique_paras.append(p)
    text = "\n\n".join(unique_paras).strip()

    text = re.sub(r"(?:#+\s*)+$", "", text).rstrip()
    if text.endswith("("):
        text = text[:-1].rstrip()
    if text.count("(") > text.count(")"):
        last = text.rfind("(")
        if last != -1:
            text = text[:last].rstrip()

    return text

# -------------------------------
# Tiny prompt cache (for identical prompts)
# -------------------------------
_CACHE: Dict[str, Tuple[float, str]] = {}
_CACHE_MAX = 128
_CACHE_TTL_SEC = 600  # 10 minutes

def _now() -> float:
    return time.time()

def _cache_key(prompt: str, params: Dict[str, Any]) -> str:
    j = json.dumps({"p": prompt, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def _cache_get(key: str) -> Optional[str]:
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, val = entry
    if _now() - ts > _CACHE_TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return val

def _cache_put(key: str, val: str) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        # drop oldest entry
        oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _CACHE.pop(oldest, None)
    _CACHE[key] = (_now(), val)

# -------------------------------
# HF client
# -------------------------------
class HuggingFaceClient:
    """
    Robust client for Hugging Face Inference Endpoints.
    - Fresh aiohttp session per call to avoid cross-event-loop issues in Streamlit.
    - Stop key fallback: try 'stop', then 'stop_sequences', then no stops (only if needed).
    - Strict context management to avoid unclosed session/connector errors.
    """
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        if not token:
            raise ValueError("HF_TOKEN is not configured.")
        if not endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured.")
        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        logger.info(f"HuggingFaceClient initialized with endpoint: {self.endpoint[:60]}...")

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        params = (parameters or MODEL_PARAMS).copy()
        # latency: conservative cap
        params["max_new_tokens"] = min(int(params.get("max_new_tokens", 512)), int(BASE_MAX_NEW_TOKENS))

        # clamp stops (≤4) if present
        def clamp(p: Dict[str, Any]) -> Dict[str, Any]:
            q = dict(p)
            for k in ("stop", "stop_sequences"):
                if k in q and isinstance(q[k], (list, tuple)):
                    q[k] = list(q[k])[:4]
            return q

        variants = []
        v1 = clamp(params)
        variants.append(("stop", v1))
        v2 = clamp(params)
        if "stop" in v2:
            v2["stop_sequences"] = v2.pop("stop")
        variants.append(("stop_sequences", v2))
        v3 = dict(params)
        for k in ("stop", "stop_sequences"):
            v3.pop(k, None)
        variants.append(("none", v3))

        last_error_text = None
        timeout = aiohttp.ClientTimeout(total=60)

        # Cache check on the primary (fast path) only
        cache_key = _cache_key(prompt, variants[0][1])
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        for i, (kind, p) in enumerate(variants, start=1):
            if i > 1:
                logger.info(f"Sending request to HF endpoint (variant {i}/3)")
            payload = {"inputs": prompt, "parameters": p}

            # Ensure strict closure via nested async-with blocks only
            try:
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.post(self.endpoint, json=payload, timeout=timeout) as resp:
                        status = resp.status
                        text = await resp.text()

                        if status != 200:
                            logger.error(f"HF endpoint returned {status}. Body: {text}")
                            last_error_text = text
                            # Validation/capability mismatch? Try next variant.
                            continue

                        try:
                            result = json.loads(text)
                        except json.JSONDecodeError:
                            out = clean_model_output(text)
                            if i == 1:
                                _cache_put(cache_key, out)
                            return out

                        if isinstance(result, list) and result and "generated_text" in result[0]:
                            generated_text = result[0]["generated_text"] or ""
                            if generated_text.startswith(prompt):
                                generated_text = generated_text[len(prompt):].lstrip()
                            out = clean_model_output(generated_text)
                            if i == 1:
                                _cache_put(cache_key, out)
                            return out

                        out = clean_model_output(str(result))
                        if i == 1:
                            _cache_put(cache_key, out)
                        return out

            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                last_error_text = str(e)
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                last_error_text = str(e)
                continue

        return f"Error: Could not connect to the model service. Last error: {last_error_text or 'unknown'}"

# -------------------------------
# Convenience wrappers
# -------------------------------
def _client() -> HuggingFaceClient:
    return HuggingFaceClient()

async def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    try:
        client = _client()
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
        "max_new_tokens": BASE_MAX_NEW_TOKENS,
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###"]
    })
    raw = await call_huggingface(prompt, base_params)
    return clean_model_output(raw)

async def call_guard_agent(prompt: str) -> str:
    guard_params = MODEL_PARAMS.copy()
    guard_params.update({
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 180,
        "stop": ["\n\n", "User:", "Assistant:", "###"]
    })
    raw = await call_huggingface(prompt, guard_params)
    return clean_model_output(raw)
