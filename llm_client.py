# llm_client.py

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
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
    r"^\s*i (?:am|'m|\'m)\s+(?:an|a)\s+ai\b",
    r"^\s*as (?:an|a) (?:ai|language model)\b",
    r"^\s*please (?:ignore|disregard)\b",
    r"^\s*for (?:educational|informational) purposes\b",
    r"^\s*this (?:message|content)\s+(?:complies|adheres|follows)\b",
    r"^\s*i cannot (?:do that|comply)\b",
    r"^\s*debug\s*:",
]

def _trim_to_complete_sentence(text: str) -> str:
    """Ensure we end on a sentence boundary; drop trailing fragments."""
    if not text:
        return text
    text = text.strip()
    if re.search(r'[.!?)]\s*$', text):
        return text
    m = list(re.finditer(r'[.!?)]', text))
    if m:
        return text[:m[-1].end()].rstrip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and len(lines[-1].split()) <= 5:
        return "\n".join(lines[:-1]).strip()
    return text

def _drop_trailing_meta_line(text: str) -> str:
    """Remove a final line like 'Refer', 'See', 'Source', etc., if it lacks punctuation."""
    if not text:
        return text
    lines = [ln.rstrip() for ln in text.splitlines()]
    if not lines:
        return text
    last = lines[-1].strip()
    if (re.match(r'^(refer|see|source|citation)\b', last, re.IGNORECASE)
        and not re.search(r'[.!?]$', last)):
        lines.pop()
        return "\n".join(lines).strip()
    return text

def _collapse_adjacent_exact_duplicates(text: str) -> str:
    """Collapse simple adjacent duplicates like 'hypertension, hypertension'."""
    return re.sub(r'\b(\w[\w-]{2,})(,?\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

def clean_model_output(text: str) -> str:
    """
    Remove prompt echo, meta/process chatter, OOB markers, duplicated paras,
    and dangling artifacts like trailing '###' or an unmatched '('.
    """
    if not text:
        return text or ""

    # Normalize & strip prompt echoes
    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()

    # Cut at common OOB markers
    for m in [
        "### End Response", "### End", "Additional Queries and Responses",
        "\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"
    ]:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx].rstrip()
            break

    # Remove parenthetical labels like (Label: …)
    text = re.sub(r"\(\s*(label|source)\s*:[^)]+\)", "", text, flags=re.IGNORECASE)

    # Drop obvious meta/process lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    cleaned_lines: List[str] = []
    for ln in lines:
        lower = ln.lower()
        if any(re.match(pat, lower) for pat in _META_LINE_PATTERNS):
            continue
        if "this response" in lower and any(k in lower for k in ["neutral", "complies", "adheres", "follows", "meets"]):
            continue
        cleaned_lines.append(ln)
    text = "\n".join(cleaned_lines).strip()

    # Deduplicate paragraph blocks
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    seen = set(); unique_paras: List[str] = []
    for p in paragraphs:
        key = p.lower()
        if key not in seen:
            seen.add(key); unique_paras.append(p)
    text = "\n\n".join(unique_paras).strip()

    # Collapse simple adjacent duplicates (comma-separated)
    text = _collapse_adjacent_exact_duplicates(text)

    # Remove trailing orphan markers (###) and dangling '('
    text = re.sub(r"(?:#+\s*)+$", "", text).rstrip()
    if text.endswith("("):
        text = text[:-1].rstrip()
    if text.count("(") > text.count(")"):
        last = text.rfind("(")
        if last != -1:
            text = text[:last].rstrip()

    # Drop trailing 'Refer/See/Source' fragments and trim to full sentence
    text = _drop_trailing_meta_line(text)
    text = _trim_to_complete_sentence(text)

    return text

def clean_streaming_chunk(chunk: str) -> str:
    """Light cleaning for streaming chunks without breaking flow"""
    # Remove obvious prompt echoes
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if chunk.startswith(prefix):
            chunk = chunk[len(prefix):].lstrip()
    return chunk


# -------------------------------
# HF client with streaming support
# -------------------------------

class HuggingFaceClient:
    """
    Enhanced client for Hugging Face Inference Endpoints with streaming support.
    - Single aiohttp session per call (reused across parameter variants).
    - Parameter fallbacks for 'stop' vs 'stop_sequences' vs none.
    - Streaming support for better perceived latency.
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
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        logger.info(f"HuggingFaceClient initialized with endpoint: {endpoint[:60]}...")

    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Standard non-streaming generation"""
        if parameters is None:
            parameters = MODEL_PARAMS.copy()

        def clamp_stops(p: Dict[str, Any]) -> Dict[str, Any]:
            p = dict(p)
            for k in ("stop", "stop_sequences"):
                if k in p and isinstance(p[k], (list, tuple)):
                    p[k] = list(p[k])[:4]
            return p

        # Build 3 parameter variants to survive endpoint differences
        variants: List[Dict[str, Any]] = []
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
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=50)

        # One session reused across variants (connection pooling, lower latency)
        async with aiohttp.ClientSession(headers=self.headers) as session:
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

                        # Expect either [{ "generated_text": "..."}] or raw text
                        try:
                            result = json.loads(text)
                        except json.JSONDecodeError:
                            return clean_model_output(text)

                        if isinstance(result, list) and result and isinstance(result[0], dict) and "generated_text" in result[0]:
                            generated_text = (result[0].get("generated_text") or "")
                            if generated_text.startswith(prompt):
                                generated_text = generated_text[len(prompt):].lstrip()
                            return clean_model_output(generated_text)

                        if isinstance(result, dict) and "generated_text" in result:
                            return clean_model_output(result.get("generated_text") or "")

                        # Fallback: stringify whatever we got
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

    async def stream_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Stream response tokens as they arrive for better perceived latency.
        Note: Requires endpoint support for streaming (not all HF endpoints support this).
        """
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        # Enable streaming in parameters
        parameters["stream"] = True
        parameters["return_full_text"] = False
        
        # Simplified parameters for streaming (avoid stop token issues)
        parameters.pop("stop", None)
        parameters.pop("stop_sequences", None)
        
        timeout = aiohttp.ClientTimeout(total=120, connect=10)  # Longer timeout for streaming
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            payload = {
                "inputs": prompt,
                "parameters": parameters,
                "stream": True
            }
            
            try:
                async with session.post(self.endpoint, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Streaming request failed: {response.status} - {error_text}")
                        yield f"Error: Streaming not supported or failed: {response.status}"
                        return
                    
                    # Process streaming response
                    accumulated = ""
                    async for line in response.content:
                        if not line:
                            continue
                        
                        try:
                            line_text = line.decode('utf-8').strip()
                            if not line_text or line_text.startswith(':'):  # Skip SSE comments
                                continue
                            
                            # Handle Server-Sent Events format
                            if line_text.startswith('data: '):
                                line_text = line_text[6:]
                            
                            if line_text == '[DONE]':
                                break
                            
                            # Parse JSON chunk
                            data = json.loads(line_text)
                            
                            # Extract token from various possible formats
                            token = None
                            if isinstance(data, dict):
                                token = data.get('token', {}).get('text', '')
                                if not token:
                                    token = data.get('generated_text', '')
                                if not token:
                                    token = data.get('text', '')
                            
                            if token:
                                # Clean and yield the token
                                clean_token = clean_streaming_chunk(token)
                                if clean_token:
                                    accumulated += clean_token
                                    yield clean_token
                                    
                        except json.JSONDecodeError:
                            # Some endpoints return plain text chunks
                            if line_text and not line_text.startswith('data:'):
                                yield clean_streaming_chunk(line_text)
                        except Exception as e:
                            logger.debug(f"Error processing stream chunk: {e}")
                            continue
                            
            except aiohttp.ClientError as e:
                logger.error(f"Streaming request failed: {e}")
                yield f"Error: Streaming failed: {str(e)}"
            except Exception as e:
                logger.error(f"Unexpected streaming error: {e}")
                yield f"Error: Unexpected streaming error: {str(e)}"


# -------------------------------
# Convenience wrappers
# -------------------------------

def _client() -> HuggingFaceClient:
    # Stateless factory—fresh client object per call is fine (session is per-call).
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

async def stream_base_assistant(prompt: str) -> AsyncGenerator[str, None]:
    """
    Stream response from base assistant for better perceived latency.
    Falls back to regular generation if streaming fails.
    """
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 300,
    })
    
    try:
        client = _client()
        async for chunk in client.stream_response(prompt, base_params):
            if chunk.startswith("Error:"):
                # Streaming failed, fall back to regular generation
                logger.warning("Streaming failed, falling back to regular generation")
                result = await call_base_assistant(prompt)
                yield result
                return
            yield chunk
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # Fall back to regular generation
        result = await call_base_assistant(prompt)
        yield result

async def call_guard_agent(prompt: str) -> str:
    guard_params = MODEL_PARAMS.copy()
    guard_params.update({
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 50,
        "stop": ["\n\n", "User:", "Assistant:", "###"]
    })
    raw = await call_huggingface(prompt, guard_params)
    return clean_model_output(raw)