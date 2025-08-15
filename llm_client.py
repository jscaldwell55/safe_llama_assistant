import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import re
import weakref

from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Output cleaning (post-generation)
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

# Inline meta sentence killers (mid-paragraph)
_INLINE_META_SENTENCES = [
    r"(?:^|\s)(?:this|the)\s+response\b[^.?!]*\b(neutral|neutrality|complies|adheres|follows|meets|is intended)\b[^.?!]*[.?!]",
    r"(?:^|\s)this\s+answer\b[^.?!]*[.?!]",
]

def _strip_inline_meta_sentences(text: str) -> str:
    out = text
    for pat in _INLINE_META_SENTENCES:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    # collapse double spaces created by removals
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

def clean_model_output(text: str) -> str:
    if not text:
        return text or ""

    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    cut_markers = [
        "### End Response", "### End", "Additional Queries and Responses",
        "\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"
    ]
    for m in cut_markers:
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
        cleaned_lines.append(ln)
    text = "\n".join(cleaned_lines).strip()

    # NEW: remove inline meta sentences even if they aren't on their own line
    text = _strip_inline_meta_sentences(text)

    # Remove trailing ### or dangling "(" artifacts
    text = re.sub(r"(?:#+\s*)+$", "", text).rstrip()
    if text.endswith("("):
        text = text[:-1].rstrip()
    if text.count("(") > text.count(")"):
        last = text.rfind("(")
        if last != -1:
            text = text[:last].rstrip()

    # Deduplicate paragraphs (after cleanup)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    seen = set()
    unique_paras = []
    for p in paragraphs:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique_paras.append(p)
    return "\n\n".join(unique_paras).strip()


# -------------------------------
# Per-event-loop session + client cache
# -------------------------------

_sessions_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, aiohttp.ClientSession]" = weakref.WeakKeyDictionary()
_clients_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, 'HuggingFaceClient']" = weakref.WeakKeyDictionary()

async def _get_loop_session(headers: Dict[str, str]) -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    sess = _sessions_by_loop.get(loop)
    if not sess or sess.closed:
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
        sess = aiohttp.ClientSession(connector=connector, headers=headers)
        _sessions_by_loop[loop] = sess
    return sess

def reset_hf_client():
    """Clear cached client and session for the current loop (e.g., after endpoint rotation)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop; clear all caches
        _clients_by_loop.clear()
        _sessions_by_loop.clear()
        return
    _clients_by_loop.pop(loop, None)
    sess = _sessions_by_loop.pop(loop, None)
    if sess and not sess.closed:
        # Close asynchronously if possible
        try:
            asyncio.create_task(sess.close())
        except RuntimeError:
            pass

def get_hf_client() -> "HuggingFaceClient":
    """Return a per-loop cached HuggingFaceClient."""
    loop = asyncio.get_running_loop()
    client = _clients_by_loop.get(loop)
    if client is None:
        client = HuggingFaceClient()
        _clients_by_loop[loop] = client
    return client


# -------------------------------
# HF client
# -------------------------------

class HuggingFaceClient:
    """
    Robust client for Hugging Face Inference Endpoints.
    - Per-event-loop session cache for connection reuse.
    - Parameter fallbacks for 'stop' vs 'stop_sequences' vs none (≤4).
    """
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        if not token:
            raise ValueError("HF_TOKEN is not configured.")
        if not endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured.")
        endpoint = endpoint.rstrip("/")
        self.token = token
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        # Make this DEBUG to reduce log spam
        logger.debug(f"HuggingFaceClient initialized with endpoint: {endpoint[:60]}...")

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        if parameters is None:
            parameters = MODEL_PARAMS.copy()

        def clamp_stops(p: Dict[str, Any]) -> Dict[str, Any]:
            p = dict(p)
            for k in ("stop", "stop_sequences"):
                if k in p and isinstance(p[k], (list, tuple)):
                    p[k] = list(p[k])[:4]
            return p

        variants: list[Dict[str, Any]] = []

        v1 = clamp_stops(parameters)
        variants.append(v1)

        v2 = clamp_stops(parameters)
        if "stop" in v2:
            v2["stop_sequences"] = v2.pop("stop")
        variants.append(v2)

        v3 = dict(parameters)
        for k in ("stop", "stop_sequences"):
            v3.pop(k, None)
        variants.append(v3)

        last_error_text = None
        timeout = aiohttp.ClientTimeout(total=60)
        session = await _get_loop_session(self.headers)

        for i, params in enumerate(variants, start=1):
            logger.debug(f"Sending request to HF endpoint (variant {i}/3)")
            payload = {"inputs": prompt, "parameters": params}

            try:
                async with session.post(self.endpoint, json=payload, timeout=timeout) as response:
                    status = response.status
                    text = await response.text()

                    if status != 200:
                        logger.error(f"HF endpoint returned {status}. Body: {text}")
                        last_error_text = text
                        continue

                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        return clean_model_output(text)

                    if isinstance(result, list) and result and isinstance(result[0], dict) and "generated_text" in result[0]:
                        generated_text = (result[0].get("generated_text") or "")
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].lstrip()
                        return clean_model_output(generated_text)

                    return clean_model_output(str(result))

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

async def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    try:
        client = get_hf_client()
        out = await client.generate_response(prompt, parameters)
        return out
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
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###"]
    })
    raw = await call_huggingface(prompt, base_params)
    return clean_model_output(raw)

async def call_guard_agent(prompt: str) -> str:
    # NOTE: Retained for compatibility; your guard.py no longer relies on LLM verdicts.
    guard_params = MODEL_PARAMS.copy()
    guard_params.update({
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 200,
        "stop": ["\n\n", "User:", "Assistant:", "###"]
    })
    raw = await call_huggingface(prompt, guard_params)
    return clean_model_output(raw)
