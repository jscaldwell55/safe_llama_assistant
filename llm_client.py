# llm_client.py
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import hashlib
import re
import time

from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Output cleaning (post-generation)
# -------------------------------

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
        # very common meta sentence variants
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


# -------------------------------
# HF client
# -------------------------------

class HuggingFaceClient:
    """
    Minimal, robust client for Hugging Face Inference Endpoints.
    - Fresh aiohttp session per call (prevents event loop issues in Streamlit).
    - Parameter fallbacks for 'stop' vs 'stop_sequences' vs none.
    """
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        if not token:
            raise ValueError("HF_TOKEN is not configured.")
        if not endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured.")

        # Normalize endpoint (strip trailing slash)
        endpoint = endpoint.rstrip("/")
        self.token = token
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        logger.info(f"HuggingFaceClient initialized with endpoint: {endpoint[:60]}...")

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        if parameters is None:
            parameters = MODEL_PARAMS.copy()

        # Ensure we never send >4 stops
        def clamp_stops(p: Dict[str, Any]) -> Dict[str, Any]:
            p = dict(p)
            for k in ("stop", "stop_sequences"):
                if k in p and isinstance(p[k], (list, tuple)):
                    p[k] = list(p[k])[:4]
            return p

        variants: list[Dict[str, Any]] = []

        # Variant 1: prefer 'stop' key if present (≤ 4)
        v1 = clamp_stops(parameters)
        variants.append(v1)

        # Variant 2: map stop -> stop_sequences, remove stop
        v2 = clamp_stops(parameters)
        if "stop" in v2:
            v2["stop_sequences"] = v2.pop("stop")
        variants.append(v2)

        # Variant 3: no stops at all
        v3 = dict(parameters)
        for k in ("stop", "stop_sequences"):
            v3.pop(k, None)
        variants.append(v3)

        last_error_text = None
        timeout = aiohttp.ClientTimeout(total=60)

        for i, params in enumerate(variants, start=1):
            logger.info(f"Sending request to HF endpoint (variant {i}/3)")
            payload = {"inputs": prompt, "parameters": params}

            # Fresh session per attempt; solves cross-loop problems
            try:
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=10, ssl=False)
                async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
                    async with session.post(self.endpoint, json=payload, timeout=timeout) as response:
                        status = response.status
                        text = await response.text()

                        if status != 200:
                            # Log the server's body to aid debugging (422, etc.)
                            logger.error(f"HF endpoint returned {status}. Body: {text}")
                            last_error_text = text
                            # try next variant
                            continue

                        # Expect either { "generated_text": "..."} list or raw text
                        try:
                            result = json.loads(text)
                        except json.JSONDecodeError:
                            return clean_model_output(text)

                        if isinstance(result, list) and result and "generated_text" in result[0]:
                            generated_text = result[0]["generated_text"] or ""
                            # Some endpoints echo the prompt; strip if needed
                            if generated_text.startswith(prompt):
                                generated_text = generated_text[len(prompt):].lstrip()
                            return clean_model_output(generated_text)

                        # Fallback: stringify
                        return clean_model_output(str(result))

            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                last_error_text = str(e)
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                last_error_text = str(e)
                continue

        # If all variants failed:
        return f"Error: Could not connect to the model service. Last error: {last_error_text or 'unknown'}"


# -------------------------------
# Convenience wrappers
# -------------------------------

# We keep a lightweight, stateless client factory. No cached session => loop-safe.
def _client() -> HuggingFaceClient:
    return HuggingFaceClient()

async def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    try:
        client = _client()
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
