# llm_client.py - Fixed output cleaning for chain-of-thought removal

import aiohttp
import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional, List, AsyncGenerator

from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# OUTPUT CLEANING - Enhanced for Chain-of-Thought Removal
# ============================================================================

def clean_model_output(text: str) -> str:
    """
    Enhanced cleaning to remove:
    - Chain-of-thought reasoning
    - Step-by-step explanations
    - Meta-commentary
    - Internal reasoning markers
    """
    if not text:
        return ""
    
    # Normalize & strip role markers
    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    
    # Remove chain-of-thought patterns
    cot_patterns = [
        # Step indicators
        r"^\*\*Step \d+:.*?\*\*\n?",
        r"^Step \d+:.*?\n",
        
        # Final answer markers
        r"The final answer is:\s*",
        r"Final response:\s*",
        r"Final answer:\s*",
        r"Here's my response:\s*",
        r"My response:\s*",
        
        # Reasoning markers
        r"Let me think.*?\n",
        r"Thinking step by step.*?\n",
        r"Breaking this down.*?\n",
        r"First,.*?Second,.*?Finally,",
        
        # Meta-commentary about the response
        r"\*\*.*?acknowledge.*?\*\*\s*",
        r"\*\*.*?express.*?\*\*\s*",
        r"I should.*?\n",
        r"I need to.*?\n",
        r"I will.*?\n",
    ]
    
    # Apply chain-of-thought removal patterns
    for pattern in cot_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Special handling for "final answer" format
    # If text contains "The final answer is:" or similar, extract only what comes after
    final_answer_match = re.search(
        r"(?:The final answer is|Final response|Final answer|Here's my response|My response):\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if final_answer_match:
        text = final_answer_match.group(1).strip()
    
    # Remove duplicate content (sometimes the model repeats the response)
    # Split by common separators and take the last clean version
    if "×" in text:
        # Special case for the × separator seen in the output
        parts = text.split("×")
        # Take the last non-empty part
        for part in reversed(parts):
            cleaned = part.strip()
            if cleaned and not any(marker in cleaned.lower() for marker in ["step", "acknowledge", "express"]):
                text = cleaned
                break
    
    # Remove any remaining step-by-step content
    lines = text.split('\n')
    cleaned_lines = []
    skip_next = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip lines that are clearly internal reasoning
        if any(marker in line_lower for marker in [
            "step 1:", "step 2:", "step 3:",
            "first:", "second:", "third:",
            "acknowledge", "express willingness",
            "i should", "i need to", "i will"
        ]):
            skip_next = True
            continue
        
        # Skip empty lines after reasoning
        if skip_next and not line.strip():
            skip_next = False
            continue
        
        skip_next = False
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines).strip()
    
    # Remove repeated content (if the same sentence appears multiple times)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if sentence_clean and sentence_clean.lower() not in seen:
            seen.add(sentence_clean.lower())
            unique_sentences.append(sentence_clean)
    
    if len(unique_sentences) < len(sentences):
        # We removed duplicates, reconstruct
        text = ' '.join(unique_sentences)
    
    # Final cleanup of common artifacts
    text = re.sub(r"(?:#+\s*)+$", "", text).rstrip()  # Remove trailing ###
    text = re.sub(r"\(\s*(label|source)\s*:[^)]+\)", "", text, flags=re.IGNORECASE)  # Remove labels
    
    # Ensure we end on a complete sentence
    if text and not text[-1] in '.!?':
        # Try to find the last complete sentence
        last_sentence = re.search(r'.*[.!?]', text)
        if last_sentence:
            text = last_sentence.group()
    
    return text.strip()

# ============================================================================
# STREAMING CHUNK CLEANING
# ============================================================================

def clean_streaming_chunk(chunk: str) -> str:
    """Light cleaning for streaming chunks"""
    # Remove obvious prompt echoes
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:"]:
        if chunk.startswith(prefix):
            chunk = chunk[len(prefix):].lstrip()
    
    # Don't do heavy processing on chunks to maintain streaming flow
    return chunk

# ============================================================================
# HUGGINGFACE CLIENT
# ============================================================================

class HuggingFaceClient:
    """Client for Hugging Face Inference Endpoints"""
    
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
        """Standard non-streaming generation with enhanced cleaning"""
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        def clamp_stops(p: Dict[str, Any]) -> Dict[str, Any]:
            p = dict(p)
            for k in ("stop", "stop_sequences"):
                if k in p and isinstance(p[k], (list, tuple)):
                    p[k] = list(p[k])[:4]
            return p
        
        # Build parameter variants
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
                        
                        # Parse response
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
    
    async def stream_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Stream response tokens for better perceived latency"""
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        parameters["stream"] = True
        parameters["return_full_text"] = False
        parameters.pop("stop", None)
        parameters.pop("stop_sequences", None)
        
        timeout = aiohttp.ClientTimeout(total=120, connect=10)
        
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
                    
                    accumulated = ""
                    async for line in response.content:
                        if not line:
                            continue
                        
                        try:
                            line_text = line.decode('utf-8').strip()
                            if not line_text or line_text.startswith(':'):
                                continue
                            
                            if line_text.startswith('data: '):
                                line_text = line_text[6:]
                            
                            if line_text == '[DONE]':
                                break
                            
                            data = json.loads(line_text)
                            
                            token = None
                            if isinstance(data, dict):
                                token = data.get('token', {}).get('text', '')
                                if not token:
                                    token = data.get('generated_text', '')
                                if not token:
                                    token = data.get('text', '')
                            
                            if token:
                                clean_token = clean_streaming_chunk(token)
                                if clean_token:
                                    accumulated += clean_token
                                    yield clean_token
                                    
                        except json.JSONDecodeError:
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

# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

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
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###", "Step 1:", "Step 2:"]
    })
    raw = await call_huggingface(prompt, base_params)
    return clean_model_output(raw)

async def stream_base_assistant(prompt: str) -> AsyncGenerator[str, None]:
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
                logger.warning("Streaming failed, falling back to regular generation")
                result = await call_base_assistant(prompt)
                yield result
                return
            yield chunk
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        result = await call_base_assistant(prompt)
        yield result

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