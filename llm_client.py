# llm_client.py
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import re
import threading

from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Output cleaning (post-generation)
# -----------------------------------------------------------------------------

# Loose patterns for one-line “meta” sentences we never want to show users
_META_LINE_PATTERNS = [
    r"^\s*note\s*:\s*",                                  # "Note: ..."
    r"^\s*this (answer|response) (?:is|was)\b",          # "This response is..."
    r"^\s*i (?:am|’m|\'m)\s+(?:an|a)\s+ai\b",            # "I'm an AI..."
    r"^\s*as (?:an|a) (?:ai|language model)\b",          # "As an AI..."
    r"^\s*please (?:ignore|disregard)\b",                # "Please ignore..."
    r"^\s*for (?:educational|informational) purposes\b", # "For informational purposes..."
    r"^\s*this (?:message|content)\s+(?:complies|adheres|follows)\b",
    r"^\s*i cannot (?:do that|comply)\b",
    r"^\s*debug\s*:",                                    # "Debug:"
]

def clean_model_output(text: str) -> str:
    """
    Remove prompt echo, meta/process chatter, OOB markers, duplicated paras,
    and dangling artifacts like trailing '###' or an unmatched '('.
    """
    if not text:
        return text or ""

    # Normalize and strip prompt echoes
    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    # Cut off at common out-of-band markers the model sometimes emits
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

    # Drop obvious meta/process lines
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

    # Remove a trailing orphan marker or dangling "(..."
    text = re.sub(r"(?:#+\s*)+$", "", text).rstrip()  # trailing ### or ####
    if text.endswith("("):
        text = text[:-1].rstrip()
    # If there are more '(' than ')', drop from last unmatched '('
    if text.count("(") > text.count(")"):
        last = text.rfind("(")
        if last != -1:
            text = text[:last].rstrip()

    return text


# -----------------------------------------------------------------------------
# HF client (single cached client per event loop)
# -----------------------------------------------------------------------------

class HuggingFaceClient:
    """
    Robust client for Hugging Face Inference Endpoints.

    - One client per asyncio event loop; the client reuses a single aiohttp session.
    - Parameter fallbacks for 'stop' vs 'stop_sequences' vs none.
    - Clean output handling.
    """
    def __init__(self, token: str, endpoint: str):
        if not token:
            raise ValueError("HF_TOKEN is not configured.")
        if not endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured.")

        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._loop_id: Optional[int] = None
        logger.info(f"HuggingFaceClient initialized with endpoint: {self.endpoint[:60]}...")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        if self._session is None or self._session.closed or self._loop_id != loop_id:
            # Create a new session bound to the current loop
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
            self._session = aiohttp.ClientSession(connector=connector, headers=self.headers)
            self._loop_id = loop_id
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._loop_id = None

    @staticmethod
    def _clamp_stops(p: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(p)
        for k in ("stop", "stop_sequences"):
            if k in p and isinstance(p[k], (list, tuple)):
                p[k] = list(p[k])[:4]
        return p

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        if parameters is None:
            parameters = MODEL_PARAMS.copy()

        # Build parameter variants
        variants: list[Dict[str, Any]] = []

        # Variant 1: prefer 'stop'
        v1 = self._clamp_stops(parameters)
        variants.append(v1)

        # Variant 2: stop -> stop_sequences
        v2 = self._clamp_stops(parameters)
        if "stop" in v2:
            v2["stop_sequences"] = v2.pop("stop")
        variants.append(v2)

        # Variant 3: no stops
        v3 = dict(parameters)
        v3.pop("stop", None)
        v3.pop("stop_sequences", None)
        variants.append(v3)

        last_error_text = None
        timeout = aiohttp.ClientTimeout(total=60)

        session = await self._ensure_session()

        for i, params in enumerate(variants, start=1):
            logger.info(f"Sending request to HF endpoint (variant {i}/3)")
            payload = {"inputs": prompt, "parameters": params}

            try:
                async with session.post(self.endpoint, json=payload, timeout=timeout) as response:
                    status = response.status
                    text = await response.text()

                    if status != 200:
                        logger.error(f"HF endpoint returned {status}. Body: {text}")
                        last_error_text = text
                        continue

                    # Parse result shape
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        return clean_model_output(text)

                    if isinstance(result, list) and result and isinstance(result[0], dict) and "generated_text" in result[0]:
                        generated_text = result[0].get("generated_text") or ""
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].lstrip()
                        return clean_model_output(generated_text)

                    # Fallback stringify
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


# -----------------------------------------------------------------------------
# Client cache & public helpers
# -----------------------------------------------------------------------------

_CLIENT_CACHE: Dict[int, HuggingFaceClient] = {}
_CLIENT_LOCK = threading.Lock()

def _current_loop_key() -> int:
    """Use the running loop id if present; otherwise 0 (non-async context)."""
    try:
        loop = asyncio.get_running_loop()
        return id(loop)
    except RuntimeError:
        return 0

def get_hf_client() -> HuggingFaceClient:
    """
    Return a cached client for the current event loop (created if missing or endpoint/token changed).
    Safe to call from sync or async code. The session is created lazily on first request.
    """
    key = _current_loop_key()
    endpoint = (HF_INFERENCE_ENDPOINT or "").rstrip("/")
    token = HF_TOKEN

    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(key)
        if client is None or client.endpoint != endpoint or client.token != token:
            client = HuggingFaceClient(token=token, endpoint=endpoint)
            _CLIENT_CACHE[key] = client
        return client

def reset_hf_client():
    """
    Close and clear all cached clients (used after endpoint rotation via UI).
    """
    with _CLIENT_LOCK:
        clients = list(_CLIENT_CACHE.values())
        _CLIENT_CACHE.clear()
    # Close sessions outside the lock (they are async)
    async def _close_all():
        for c in clients:
            try:
                await c.close()
            except Exception:
                pass
    # Try to run quickly whether we're in a loop or not
    try:
        loop = asyncio.get_running_loop()
        # Fire and forget; sessions will close shortly
        loop.create_task(_close_all())
    except RuntimeError:
        asyncio.run(_close_all())


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
        # Try to stop on role markers / section fences; we cap at 4 internally
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
        "max_new_tokens": 200,
        "stop": ["\n\n", "User:", "Assistant:", "###"]
    })
    raw = await call_huggingface(prompt, guard_params)
    return clean_model_output(raw)
