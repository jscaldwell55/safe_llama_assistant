# llm_client.py - Complete Version with All Fixes

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
# RETRY CONFIGURATION
# ============================================================================

MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0  # Base delay in seconds
RETRY_DELAY_MAX = 10.0  # Maximum delay in seconds

# ============================================================================
# REQUEST BATCHING FOR A10G EFFICIENCY
# ============================================================================

@dataclass
class BatchRequest:
    prompt: str
    parameters: Dict[str, Any]
    future: asyncio.Future
    timestamp: float
    retry_count: int = 0

class RequestBatcher:
    """Batch multiple requests for efficient A10G utilization"""
    
    def __init__(self, max_batch_size: int = 4, timeout_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests: deque = deque()
        self.batch_lock = asyncio.Lock()
        self.batch_task = None
        self._closed = False
    
    async def add_request(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Add request to batch and wait for result"""
        if self._closed:
            raise RuntimeError("RequestBatcher is closed")
            
        future = asyncio.Future()
        request = BatchRequest(prompt, parameters, future, time.time())
        
        async with self.batch_lock:
            self.pending_requests.append(request)
            
            # Start batch processor if not running
            if self.batch_task is None or self.batch_task.done():
                self.batch_task = asyncio.create_task(self._process_batch())
            
            # If batch is full, process immediately
            if len(self.pending_requests) >= self.max_batch_size:
                if self.batch_task and not self.batch_task.done():
                    self.batch_task.cancel()
                self.batch_task = asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process pending requests as a batch"""
        if self._closed:
            return
            
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
        
        # Process in parallel with retry logic
        tasks = [
            self._process_single_with_retry(req)
            for req in batch
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for req, result in zip(batch, results):
                if isinstance(result, Exception):
                    if not req.future.done():
                        req.future.set_exception(result)
                else:
                    if not req.future.done():
                        req.future.set_result(result)
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def _process_single_with_retry(self, request: BatchRequest) -> str:
        """Process a single request with retry logic"""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                client = await get_singleton_client()
                result = await client.generate_response(request.prompt, request.parameters)
                
                # Check if result is an error
                if result.startswith("Error:") and attempt < MAX_RETRIES - 1:
                    raise Exception(result)
                    
                return result
                
            except Exception as e:
                last_error = e
                request.retry_count = attempt + 1
                
                if attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_DELAY_BASE * (2 ** attempt), RETRY_DELAY_MAX)
                    logger.warning(f"Request retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {MAX_RETRIES} attempts: {str(e)[:100]}")
        
        # Return error message if all retries failed
        return f"Error: Request failed after {MAX_RETRIES} attempts - {str(last_error)}"
    
    async def close(self):
        """Close the batcher and cancel pending tasks"""
        self._closed = True
        
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()
            
        # Cancel all pending requests
        async with self.batch_lock:
            while self.pending_requests:
                req = self.pending_requests.popleft()
                if not req.future.done():
                    req.future.cancel()

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
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:", "Bot:", "AI:", "Model:", "Response:"]:
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
        r"^Let's think about this.*?\n",
        
        # Final answer patterns
        r"^(?:The )?[Ff]inal (?:answer|response) is:\s*",
        r"^Here'?s? (?:my|the) response:\s*",
        r"^My (?:response|answer):\s*",
        r"^Based on the (?:context|documentation):\s*",
        
        # Meta-commentary
        r"\*\*.*?acknowledge.*?\*\*\s*",
        r"\*\*.*?express.*?\*\*\s*",
        r"^\[.*?\]\s*",  # Remove bracketed meta text
        r"^Note:.*?\n",
        r"^Important:.*?\n",
        r"^Remember:.*?\n",
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
        "The answer is:",
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
                for keyword in ["step", "acknowledge", "express", "should", "need to", "let me", "first"]
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
    text = re.sub(r"\[\s*(?:label|source|note)\s*:[^\]]+\]", "", text, flags=re.IGNORECASE)
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
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
    A10G-optimized client with proper session management and retry logic
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
        
        # Session management
        self.session = None
        self.session_lock = asyncio.Lock()
        self._closed = False
        
        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
        
        logger.info(f"HuggingFaceClient initialized for A10G: {endpoint[:60]}...")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the session with proper locking"""
        if self._closed:
            raise RuntimeError("HuggingFaceClient is closed")
            
        async with self.session_lock:
            if self.session is None or self.session.closed:
                # Connection pool configuration for A10G
                connector = aiohttp.TCPConnector(
                    limit=10,  # Total connection pool limit
                    limit_per_host=5,  # Per-host limit
                    ttl_dns_cache=300,  # DNS cache timeout
                    enable_cleanup_closed=True,
                    force_close=False,  # Keep connections alive
                    keepalive_timeout=30,  # Keep connections alive for 30s
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=None,  # No total timeout (handled per-request)
                    connect=5,  # Connection timeout
                    sock_connect=5,  # Socket connection timeout
                    sock_read=None  # No socket read timeout (handled per-request)
                )
                
                self.session = aiohttp.ClientSession(
                    headers=self.headers,
                    connector=connector,
                    timeout=timeout
                )
                logger.info("Created new aiohttp session")
            
            return self.session
    
    async def close(self):
        """Close the session properly"""
        self._closed = True
        
        async with self.session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
                logger.info(f"Closed aiohttp session. Stats: {self.request_count} requests, {self.error_count} errors, avg latency: {self.total_latency/max(1, self.request_count):.0f}ms")
    
    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Optimized generation for A10G with retry logic
        """
        if self._closed:
            return "Error: Client is closed"
            
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        start_time = time.time()
        self.request_count += 1
        
        # Request-specific timeout
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total request timeout
            sock_read=25  # Socket read timeout
        )
        
        # Get the shared session
        try:
            session = await self._get_session()
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            self.error_count += 1
            return f"Error: Failed to initialize connection"
        
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
                    logger.info("Model loading, waiting 2s...")
                    await asyncio.sleep(2)
                    return await self.generate_response(prompt, parameters)
                
                text = await response.text()
                
                if status != 200:
                    logger.error(f"HF endpoint returned {status}: {text[:200]}")
                    self.error_count += 1
                    
                    # Handle specific error codes
                    if status == 429:  # Rate limit
                        return "Error: Rate limit exceeded. Please wait a moment."
                    elif status == 500:  # Server error
                        return "Error: Server error. Please try again."
                    else:
                        return f"Error: Service returned status {status}"
                
                # Parse response
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    # Sometimes the response is plain text
                    cleaned = clean_model_output(text)
                    if cleaned:
                        return cleaned
                    logger.warning("Failed to parse JSON response, text was empty after cleaning")
                    return "Error: Invalid response format"
                
                # Extract generated text
                generated = ""
                if isinstance(result, list) and result:
                    generated = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    generated = result.get("generated_text", "")
                    if not generated:
                        # Try alternative field names
                        generated = result.get("text", "") or result.get("output", "")
                else:
                    generated = str(result)
                
                if not generated:
                    logger.warning("No generated text in response")
                    return "Error: No response generated"
                
                # Remove prompt from response if present
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].lstrip()
                
                # Log performance
                elapsed_ms = int((time.time() - start_time) * 1000)
                self.total_latency += elapsed_ms
                
                if elapsed_ms > 5000:
                    logger.warning(f"[PERF] Slow generation: {elapsed_ms}ms")
                else:
                    logger.info(f"[PERF] Generation completed in {elapsed_ms}ms")
                
                return clean_model_output(generated)
                
        except asyncio.TimeoutError:
            logger.error("Request timed out after 30s")
            self.error_count += 1
            return "Error: Request timed out. Please try again."
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}")
            self.error_count += 1
            return f"Error: Connection failed - {str(e)[:100]}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self.error_count += 1
            return "Error: An unexpected error occurred."
    
    async def stream_response(
        self,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming for A10G - better perceived latency
        """
        if self._closed:
            yield "Error: Client is closed"
            return
            
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        parameters["stream"] = True
        parameters["return_full_text"] = False
        
        timeout = aiohttp.ClientTimeout(total=60, sock_read=30)
        
        try:
            session = await self._get_session()
        except Exception as e:
            logger.error(f"Failed to get session for streaming: {e}")
            yield "Error: Failed to initialize streaming"
            return
        
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
                    logger.error(f"Streaming failed with status {response.status}: {error_text[:200]}")
                    yield f"Error: Streaming not available (status {response.status})"
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
                                    yield clean_model_output(token)
                            except json.JSONDecodeError:
                                continue
                                
                    except UnicodeDecodeError:
                        continue
                        
        except asyncio.TimeoutError:
            logger.error("Streaming timed out")
            yield "Error: Streaming timed out"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: Streaming failed - {str(e)[:100]}"

# ============================================================================
# SINGLETON CLIENT MANAGEMENT
# ============================================================================

_client_instance: Optional[HuggingFaceClient] = None
_client_lock = asyncio.Lock()

async def get_singleton_client() -> HuggingFaceClient:
    """Get or create singleton client instance with proper locking"""
    global _client_instance
    
    async with _client_lock:
        if _client_instance is None or _client_instance._closed:
            _client_instance = HuggingFaceClient()
        return _client_instance

# ============================================================================
# CLEANUP HANDLER
# ============================================================================

async def cleanup():
    """Cleanup function to be called on app shutdown"""
    global _client_instance, request_batcher
    
    logger.info("Starting cleanup...")
    
    # Close the batcher
    if request_batcher:
        await request_batcher.close()
        logger.info("Closed request batcher")
    
    # Close the client
    if _client_instance:
        await _client_instance.close()
        _client_instance = None
    
    logger.info("Cleanup completed")

# ============================================================================
# OPTIMIZED WRAPPER FUNCTIONS WITH RETRY
# ============================================================================

async def call_huggingface_with_retry(
    prompt: str,
    parameters: Optional[Dict[str, Any]] = None,
    max_retries: int = MAX_RETRIES
) -> str:
    """Call HuggingFace with automatic retry on failure"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = await call_huggingface(prompt, parameters)
            
            # Check if result is an error and retry if needed
            if result.startswith("Error:") and "timed out" in result.lower() and attempt < max_retries - 1:
                raise Exception(result)
            
            return result
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(RETRY_DELAY_BASE * (2 ** attempt), RETRY_DELAY_MAX)
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {str(e)[:100]}")
                await asyncio.sleep(delay)
    
    return f"Error: Failed after {max_retries} attempts - {str(last_error)}"

async def call_huggingface(
    prompt: str,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Main entry point with optional batching
    """
    try:
        # Use batching if enabled and appropriate
        if request_batcher and ENABLE_REQUEST_BATCHING and not request_batcher._closed:
            if parameters is None or parameters.get("max_new_tokens", 0) <= 100:
                # Use batching for small requests
                return await request_batcher.add_request(prompt, parameters or MODEL_PARAMS)
        
        # Direct call for large requests or when batching disabled
        client = await get_singleton_client()
        return await client.generate_response(prompt, parameters)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return f"Error: Configuration issue - {str(e)[:100]}"
    except Exception as e:
        logger.error(f"Unexpected error in call_huggingface: {e}", exc_info=True)
        return "Error: An unexpected error occurred."

async def call_base_assistant(prompt: str) -> str:
    """Optimized base assistant call"""
    from config import MODEL_PARAMS
    
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 200,
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###", "Step "]
    })
    
    return await call_huggingface_with_retry(prompt, base_params)

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
        client = await get_singleton_client()
        async for chunk in client.stream_response(prompt, base_params):
            if chunk.startswith("Error:"):
                # Fall back to regular generation
                result = await call_base_assistant(prompt)
                yield result
                return
            yield chunk
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
    return await call_huggingface_with_retry(prompt, EMPATHETIC_COMPANION_PARAMS)

async def call_information_navigator(prompt: str) -> str:
    """Factual information extraction"""
    from config import INFORMATION_NAVIGATOR_PARAMS
    return await call_huggingface_with_retry(prompt, INFORMATION_NAVIGATOR_PARAMS)

async def call_bridge_synthesizer(prompt: str) -> str:
    """Response synthesis"""
    from config import BRIDGE_SYNTHESIZER_PARAMS
    return await call_huggingface_with_retry(prompt, BRIDGE_SYNTHESIZER_PARAMS)