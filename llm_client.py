# llm_client.py
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import re

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
# HF client
# -------------------------------

class HuggingFaceClient:
    """
    Robust client for Hugging Face Inference Endpoints.
    - Fresh aiohttp session per call (loop-safe in Streamlit).
    - Strict session/connector cleanup even on cancellations/hot-reload.
    - Parameter fallbacks for 'stop' vs 'stop_sequences' vs none.
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
        logger.info(f"HuggingFaceClient initialized with endpoint: {endpoint[:60]}...")

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
        timeout = aiohttp.ClientTimeout(total=60, sock_connect=20, sock_read=40)

        for i, params in enumerate(variants, start=1):
            logger.info(f"Sending request to HF endpoint (variant {i}/3)")
            payload = {"inputs": prompt, "parameters": params}

            session: Optional[aiohttp.ClientSession] = None
            connector: Optional[aiohttp.TCPConnector] = None

            try:
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
                session = aiohttp.ClientSession(connector=connector, headers=self.headers)

                async with session.post(self.endpoint, json=payload, timeout=timeout) as response:
                    status = response.status
                    text = await response.text()

                    if status != 200:
                        logger.error(f"HF endpoint returned {status}. Body: {text}")
                        last_error_text = text
                        # try next variant
                        continue

                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        return clean_model_output(text)

                    if isinstance(result, list) and result and isinstance(result[0], dict) and "generated_text" in result[0]:
                        generated_text = result[0].get("generated_text") or ""
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].lstrip()
                        return clean_model_output(generated_text)

                    return clean_model_output(str(result))

            except asyncio.CancelledError:
                # Hot-reload or task cancellation—ensure cleanup, then re-raise
                try:
                    if session and not session.closed:
                        await session.close()
                finally:
                    if connector:
                        try:
                            await connector.close()
                        except Exception:
                            pass
                raise
            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                last_error_text = str(e)
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                last_error_text = str(e)
                continue
            finally:
                # Ensure we always close on every attempt (even on early return, finally still runs)
                try:
                    if session and not session.closed:
                        await session.close()
                except Exception:
                    pass
                try:
                    if connector:
                        await connector.close()
                except Exception:
                    pass

        return f"Error: Could not connect to the model service. Last error: {last_error_text or 'unknown'}"


# -------------------------------
# Convenience wrappers
# -------------------------------

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
