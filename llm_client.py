# llm_client.py - A10G Optimized with Request Batching and Better Cleaning

import aiohttp
import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
from collections import deque

from config import (
    HF_TOKEN, 
    HF_INFERENCE_ENDPOINT, 
    MODEL_PARAMS,
    ENABLE_REQUEST_BATCHING,
    BATCH_TIMEOUT_MS,
    MAX_BATCH_SIZE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST BATCHING FOR A10G EFFICIENCY
# ============================================================================

@dataclass
class BatchRequest:
    prompt: str
    parameters: Dict[str, Any]
    future: asyncio.Future
    timestamp: float

class RequestBatcher:
    """Batch multiple requests for efficient A10G utilization"""
    
    def __init__(self, max_batch_size: int = 4, timeout_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests: deque = deque()
        self.batch_lock = asyncio.Lock()
        self.batch_task = None
    
    async def add_request(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Add request to batch and wait for result"""
        future = asyncio.Future()
        request = BatchRequest(prompt, parameters, future, time.time())
        
        async with self.batch_lock:
            self.pending_requests.append(request)
            
            # Start batch processor if not running
            if self.batch_task is None or self.batch_task.done():
                self.batch_task = asyncio.create_task(self._process_batch())
            
            # If batch is full, process immediately
            if len(self.pending_requests) >= self.max_batch_size:
                self.batch_task.cancel()
                self.batch_task = asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process pending requests as a batch"""
        # Wait for timeout or batch to fill
        await asyncio.sleep(self.timeout_ms / 1000)
        
        async with self.batch_lock:
            if not self.pending_requests:
                return
            
            # Take up to max_batch_size requests
            batch = []
            for _ in range(min(self.max_batch_size, len(self.pending_requests))):
                batch.append(self.pending_requests.popleft())
            
            if not batch:
                return
        
        # Process batch
        logger.info(f"[BATCH] Processing {len(batch)} requests")
        
        # For now, process in parallel (future: true batched inference)
        tasks = [
            self._process_single(req.prompt, req.parameters)
            for req in batch
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for req, result in zip(batch, results):
                if isinstance(result, Exception):
                    req.future.set_exception(result)
                else:
                    req.future.set_result(result)
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def _process_single(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Process a single request (placeholder for actual implementation)"""
        # This will be replaced with actual HF client call
        client = HuggingFaceClient()
        return await client.generate_response(prompt, parameters)

# Global batcher instance
request_batcher = RequestBatcher() if ENABLE_REQUEST_BATCHING else None

# ============================================================================
# ENHANCED OUTPUT CLEANING
# ============================================================================

def clean_model_output(text: str) -> str:
    """
    Enhanced cleaning for A10G responses with chain-of-thought removal
    """
    if not text:
        return ""
    
    # Remove role markers
    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:", "Bot:", "AI:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    
    # Remove chain-of-thought patterns
    cot_patterns = [
        # Step patterns
        r"^\*\*Step \d+:.*?\*\*\s*",
        r"^Step \d+:.*?\n",
        r"^\d+\.\s*\*\*.*?\*\*\s*",
        
        # Thinking patterns
        r"^Let me.*?:\s*",
        r"^I need to.*?:\s*",
        r"^First,.*?Second,.*?Finally,\s*",
        r"^Thinking step by step.*?\n",
        
        # Final answer patterns
        r"^(?:The )?[Ff]inal (?:answer|response) is:\s*",
        r"^Here'?s? (?:my|the) response:\s*",
        r"^My (?:response|answer):\s*",
        
        # Meta-commentary
        r"\*\*.*?acknowledge.*?\*\*\s*",
        r"\*\*.*?express.*?\*\*\s*",
        r"^\[.*?\]\s*",  # Remove bracketed meta text
        r"^Note:.*?\n",
        r"^Important:.*?\n",
    ]
    
    for pattern in cot_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Extract content after "final answer" markers
    final_markers = [
        "The final answer is:",
        "Final response:",
        "Final answer:",
        "Here's my response:",
        "My response is:",
    ]
    
    for marker in final_markers:
        if marker.lower() in text.lower():
            parts = re.split(re.escape(marker), text, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[-1].strip()
                break
    
    # Handle special separators
    if "×" in text:
        parts = text.split("×")
        for part in reversed(parts):
            cleaned = part.strip()
            if cleaned and not any(
                keyword in cleaned.lower() 
                for keyword in ["step", "acknowledge", "express", "should", "need to"]
            ):
                text = cleaned
                break
    
    # Remove duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for sent in sentences:
        sent_clean = sent.strip().lower()
        if sent_clean and sent_clean not in seen:
            seen.add(sent_clean)
            unique.append(sent.strip())
    
    if len(unique) < len(sentences):
        text = ' '.join(unique)
    
    # Clean up artifacts
    text = re.sub(r"(?:#+\s*)+$", "", text).rstrip()
    text = re.sub(r"\(\s*(?:label|source|note)\s*:[^)]+\)", "", text, flags=re.IGNORECASE)
    
    # Ensure complete sentence
    if text and text[-1] not in '.!?':
        match = re.search(r'.*[.!?]', text)
        if match:
            text = match.group()
    
    return text.strip()

# ============================================================================
# A10G OPTIMIZED HUGGINGFACE CLIENT
# ============================================================================

class HuggingFaceClient:
    """
    A10G-optimized client with connection pooling and better error handling
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
        
        # Connection pool configuration for A10G
        self.connector = aiohttp.TCPConnector(
            limit=10,  # Total connection pool limit
            limit_per_host=5,  # Per-host limit
            ttl_dns_cache=300,  # DNS cache timeout
            enable_cleanup_closed=True
        )
        
        logger.info(f"HuggingFaceClient initialized for A10G: {endpoint[:60]}...")
    
    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Optimized generation for A10G with better timeout handling
        """
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        start_time = time.time()
        
        # A10G can handle longer timeouts reliably
        timeout = aiohttp.ClientTimeout(
            total=30,  # Reduced from 60
            connect=5,  # Faster connect timeout
            sock_read=25  # Read timeout
        )
        
        # Single session with connection reuse
        async with aiohttp.ClientSession(
            headers=self.headers,
            connector=self.connector
        ) as session:
            payload = {
                "inputs": prompt,
                "parameters": parameters,
                "options": {
                    "use_cache": True,  # Enable KV cache on A10G
                    "wait_for_model": False  # Don't wait if model needs loading
                }
            }
            
            try:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    timeout=timeout
                ) as response:
                    status = response.status
                    
                    if status == 503:  # Model loading
                        logger.info("Model loading, retrying...")
                        await asyncio.sleep(2)
                        return await self.generate_response(prompt, parameters)
                    
                    text = await response.text()
                    
                    if status != 200:
                        logger.error(f"HF endpoint returned {status}: {text[:200]}")
                        return f"Error: Service returned status {status}"
                    
                    # Parse response
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        return clean_model_output(text)
                    
                    # Extract generated text
                    if isinstance(result, list) and result:
                        generated = result[0].get("generated_text", "")
                    elif isinstance(result, dict):
                        generated = result.get("generated_text", "")
                    else:
                        generated = str(result)
                    
                    # Remove prompt from response if present
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].lstrip()
                    
                    # Log performance
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    logger.info(f"[PERF] Generation completed in {elapsed_ms}ms")
                    
                    return clean_model_output(generated)
                    
            except asyncio.TimeoutError:
                logger.error("Request timed out")
                return "Error: Request timed out. Please try again."
            except aiohttp.ClientError as e:
                logger.error(f"Client error: {e}")
                return f"Error: Connection failed - {str(e)}"
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                return "Error: An unexpected error occurred."
    
    async def stream_response(
        self,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming for A10G - better perceived latency
        """
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        parameters["stream"] = True
        parameters["return_full_text"] = False
        
        timeout = aiohttp.ClientTimeout(total=60, connect=5)
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            connector=self.connector
        ) as session:
            payload = {
                "inputs": prompt,
                "parameters": parameters,
                "stream": True,
                "options": {"use_cache": True}
            }
            
            try:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Streaming failed: {response.status}")
                        yield f"Error: Streaming not available"
                        return
                    
                    buffer = ""
                    async for chunk in response.content.iter_chunked(1024):
                        if not chunk:
                            continue
                        
                        try:
                            text = chunk.decode('utf-8')
                            buffer += text
                            
                            # Process complete lines
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                
                                if not line or line.startswith(':'):
                                    continue
                                
                                if line.startswith('data: '):
                                    line = line[6:]
                                
                                if line == '[DONE]':
                                    return
                                
                                try:
                                    data = json.loads(line)
                                    token = (
                                        data.get('token', {}).get('text', '') or
                                        data.get('generated_text', '') or
                                        data.get('text', '')
                                    )
                                    
                                    if token:
                                        yield token
                                except json.JSONDecodeError:
                                    continue
                                    
                        except UnicodeDecodeError:
                            continue
                            
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"Error: Streaming failed"

# ============================================================================
# OPTIMIZED WRAPPER FUNCTIONS
# ============================================================================

# Cache for client instance
_client_instance: Optional[HuggingFaceClient] = None

def _get_client() -> HuggingFaceClient:
    """Get or create client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = HuggingFaceClient()
    return _client_instance

async def call_huggingface(
    prompt: str,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Main entry point with optional batching
    """
    try:
        # Use batching if enabled and appropriate
        if request_batcher and ENABLE_REQUEST_BATCHING:
            if parameters is None or parameters.get("max_new_tokens", 0) < 100:
                # Use batching for small requests
                return await request_batcher.add_request(prompt, parameters or MODEL_PARAMS)
        
        # Direct call for large requests or when batching disabled
        client = _get_client()
        return await client.generate_response(prompt, parameters)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return f"Configuration error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return "Error: An unexpected error occurred."

async def call_base_assistant(prompt: str) -> str:
    """Optimized base assistant call"""
    from config import MODEL_PARAMS
    
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 200,  # Reduced for speed
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###", "Step "]
    })
    
    return await call_huggingface(prompt, base_params)

async def stream_base_assistant(prompt: str) -> AsyncGenerator[str, None]:
    """Streaming assistant for better perceived latency"""
    from config import MODEL_PARAMS
    
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 200,
    })
    
    try:
        client = _get_client()
        async for chunk in client.stream_response(prompt, base_params):
            if chunk.startswith("Error:"):
                # Fall back to regular generation
                result = await call_base_assistant(prompt)
                yield result
                return
            yield clean_model_output(chunk)
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        result = await call_base_assistant(prompt)
        yield result

async def call_guard_agent(prompt: str) -> str:
    """Optimized guard agent call"""
    from config import GUARD_MODEL_PARAMS
    
    return await call_huggingface(prompt, GUARD_MODEL_PARAMS)

# ============================================================================
# SPECIALIZED PERSONA CALLS (A10G Optimized)
# ============================================================================

async def call_intent_classifier(prompt: str) -> str:
    """Ultra-fast intent classification"""
    from config import INTENT_CLASSIFIER_PARAMS
    return await call_huggingface(prompt, INTENT_CLASSIFIER_PARAMS)

async def call_empathetic_companion(prompt: str) -> str:
    """Empathetic response generation"""
    from config import EMPATHETIC_COMPANION_PARAMS
    return await call_huggingface(prompt, EMPATHETIC_COMPANION_PARAMS)

async def call_information_navigator(prompt: str) -> str:
    """Factual information extraction"""
    from config import INFORMATION_NAVIGATOR_PARAMS
    return await call_huggingface(prompt, INFORMATION_NAVIGATOR_PARAMS)

async def call_bridge_synthesizer(prompt: str) -> str:
    """Response synthesis"""
    from config import BRIDGE_SYNTHESIZER_PARAMS
    return await call_huggingface(prompt, BRIDGE_SYNTHESIZER_PARAMS)