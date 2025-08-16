# llm_client.py - Final Version with Event Loop & Output Cleaning Fixes

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
# EVENT LOOP MANAGEMENT FOR STREAMLIT
# ============================================================================

def get_or_create_event_loop():
    """Get the current event loop or create a new one if needed"""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

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
            
        loop = get_or_create_event_loop()
        future = loop.create_future()
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
                if result.startswith("Error:") and "Event loop" not in result and attempt < MAX_RETRIES - 1:
                    raise Exception(result)
                    
                return result
                
            except Exception as e:
                last_error = e
                request.retry_count = attempt + 1
                
                # If it's an event loop error, recreate the client
                if "Event loop" in str(e) or "Session is closed" in str(e):
                    logger.warning("Event loop/session issue detected, resetting client")
                    await reset_singleton_client()
                
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
# ENHANCED OUTPUT CLEANING - FIXED FOR PROPER FORMATTING
# ============================================================================

def clean_model_output(text: str) -> str:
    """
    Enhanced cleaning for A10G responses - removes extraction artifacts and formats naturally
    """
    if not text:
        return ""
    
    original_text = text  # Keep original for debugging
    
    # Remove extraction format artifacts FIRST
    text = re.sub(r'\*\*Extracted Information:\*\*\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Extracted Information:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^\*\*.*?Information.*?:\*\*\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove role markers
    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:", "Bot:", "AI:", "Model:", "Response:", "Navigator:", "Information Navigator:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    
    # Convert bullet points to natural text
    has_bullets = '•' in text or re.search(r'^\s*[-*]\s+', text, re.MULTILINE)
    
    if has_bullets:
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Remove bullet markers
            line = re.sub(r'^[•\-\*]\s*', '', line)
            # Skip headers
            if line and not line.endswith(':'):
                # Ensure line ends with period if it doesn't have punctuation
                if line and line[-1] not in '.!?':
                    line += '.'
                cleaned_lines.append(line)
        
        # Join facts with better flow
        if cleaned_lines:
            # Group related facts
            result = []
            for i, line in enumerate(cleaned_lines):
                if i == 0:
                    result.append(line)
                elif 'mg' in line.lower() or 'tablet' in line.lower() or 'dose' in line.lower():
                    # Group dosage information
                    result.append(f" {line}")
                else:
                    result.append(f" {line}")
            
            text = ''.join(result)
    
    # Remove chain-of-thought patterns
    cot_patterns = [
        r"^\*\*Step \d+:.*?\*\*\s*",
        r"^Step \d+:.*?\n",
        r"^Let me.*?:\s*",
        r"^I need to.*?:\s*",
        r"^Based on the (?:context|documentation):\s*",
        r"^According to the documentation:\s*",
        r"^From the documentation:\s*",
        r"^The documentation states:\s*",
        r"^Here's what I found:\s*",
        r"^Here is the information:\s*",
        r"^\*\*.*?acknowledge.*?\*\*\s*",
        r"^\[.*?\]\s*",
        r"^Note:.*?\n",
    ]
    
    for pattern in cot_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up formatting artifacts
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markdown
    text = re.sub(r'__(.*?)__', r'\1', text)  # Remove underline markdown
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*', r'\1', text)  # Remove single asterisk italics
    
    # Fix spacing issues
    text = re.sub(r'\n\s*\n', ' ', text)  # Multiple newlines to space
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = text.strip()
    
    # Ensure natural flow
    # Fix common issues like "base. The" -> "base. The"
    text = re.sub(r'([a-z])\.\s*([A-Z])', r'\1. \2', text)
    
    # Ensure complete sentence
    if text and text[-1] not in '.!?':
        text += '.'
    
    # Log if significant cleaning happened
    if len(original_text) - len(text) > 50:
        logger.debug(f"Significant cleaning: {len(original_text)} -> {len(text)} chars")
    
    return text.strip()

# ============================================================================
# A10G OPTIMIZED HUGGINGFACE CLIENT
# ============================================================================

class HuggingFaceClient:
    """
    A10G-optimized client with proper session and event loop management
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
        self._loop = None  # Track the loop the session was created with
        
        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
        
        logger.info(f"HuggingFaceClient initialized for A10G: {endpoint[:60]}...")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the session with proper event loop handling"""
        if self._closed:
            raise RuntimeError("HuggingFaceClient is closed")
        
        current_loop = get_or_create_event_loop()
        
        async with self.session_lock:
            # Check if we need a new session (loop changed or session closed)
            if (self.session is None or 
                self.session.closed or 
                self._loop != current_loop or
                (self._loop and self._loop.is_closed())):
                
                # Close old session if it exists
                if self.session and not self.session.closed:
                    try:
                        await self.session.close()
                    except Exception as e:
                        logger.warning(f"Error closing old session: {e}")
                
                # Create new connector for this loop
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                    force_close=False,
                    keepalive_timeout=30,
                    loop=current_loop  # Explicitly set the loop
                )
                
                # Create new session with current loop
                self.session = aiohttp.ClientSession(
                    headers=self.headers,
                    connector=connector,
                    loop=current_loop  # Explicitly set the loop
                )
                self._loop = current_loop
                logger.info("Created new aiohttp session with current event loop")
            
            return self.session
    
    async def close(self):
        """Close the session properly"""
        self._closed = True
        
        async with self.session_lock:
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                    await asyncio.sleep(0.1)  # Give it time to close properly
                except Exception as e:
                    logger.warning(f"Error during session close: {e}")
                self.session = None
                self._loop = None
                logger.info(f"Closed aiohttp session. Stats: {self.request_count} requests, {self.error_count} errors")
    
    async def reset(self):
        """Reset the client by closing and allowing recreation of session"""
        logger.info("Resetting HuggingFaceClient")
        await self.close()
        self._closed = False
        self._loop = None
        self.session = None
    
    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Optimized generation for A10G with better error handling
        """
        if self._closed:
            # Try to reset if closed
            await self.reset()
            
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        start_time = time.time()
        self.request_count += 1
        
        # Request-specific timeout
        timeout = aiohttp.ClientTimeout(
            total=30,
            sock_read=25
        )
        
        try:
            # Get the session (will create new one if needed)
            session = await self._get_session()
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            self.error_count += 1
            # Try to reset and get session again
            await self.reset()
            try:
                session = await self._get_session()
            except Exception as e2:
                logger.error(f"Failed to get session after reset: {e2}")
                return "Error: Failed to initialize connection"
        
        payload = {
            "inputs": prompt,
            "parameters": parameters,
            "options": {
                "use_cache": True,
                "wait_for_model": False
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
                    
                    if status == 429:
                        return "Error: Rate limit exceeded. Please wait a moment."
                    elif status == 500:
                        return "Error: Server error. Please try again."
                    else:
                        return f"Error: Service returned status {status}"
                
                # Parse response
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    cleaned = clean_model_output(text)
                    if cleaned:
                        return cleaned
                    logger.warning("Failed to parse JSON response")
                    return "Error: Invalid response format"
                
                # Extract generated text
                generated = ""
                if isinstance(result, list) and result:
                    generated = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    generated = result.get("generated_text", "")
                    if not generated:
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
            # Reset on connection errors
            if "Cannot connect" in str(e) or "Connection reset" in str(e):
                await self.reset()
            return f"Error: Connection failed - {str(e)[:100]}"
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.error("Event loop was closed, resetting client")
                await self.reset()
                return "Error: Event loop issue. Please try again."
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self.error_count += 1
            return "Error: An unexpected error occurred."

# ============================================================================
# SINGLETON CLIENT MANAGEMENT
# ============================================================================

_client_instance: Optional[HuggingFaceClient] = None
_client_lock = asyncio.Lock()

async def get_singleton_client() -> HuggingFaceClient:
    """Get or create singleton client instance with proper locking"""
    global _client_instance
    
    try:
        async with _client_lock:
            if _client_instance is None or _client_instance._closed:
                _client_instance = HuggingFaceClient()
            return _client_instance
    except RuntimeError as e:
        if "Event loop" in str(e):
            # Create with new event loop
            loop = get_or_create_event_loop()
            async with _client_lock:
                _client_instance = HuggingFaceClient()
                return _client_instance
        raise

async def reset_singleton_client():
    """Reset the singleton client"""
    global _client_instance
    
    async with _client_lock:
        if _client_instance:
            await _client_instance.reset()

# ============================================================================
# CLEANUP HANDLER
# ============================================================================

async def cleanup():
    """Cleanup function to be called on app shutdown"""
    global _client_instance, request_batcher
    
    logger.info("Starting cleanup...")
    
    # Close the batcher
    if request_batcher:
        try:
            await request_batcher.close()
            logger.info("Closed request batcher")
        except Exception as e:
            logger.warning(f"Error closing batcher: {e}")
    
    # Close the client
    if _client_instance:
        try:
            await _client_instance.close()
            _client_instance = None
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
    
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
            if result.startswith("Error:") and "Event loop" in result and attempt < max_retries - 1:
                logger.warning("Event loop error detected, resetting client")
                await reset_singleton_client()
                await asyncio.sleep(0.5)
                continue
            
            return result
            
        except Exception as e:
            last_error = e
            if "Event loop" in str(e):
                await reset_singleton_client()
            
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
                try:
                    return await request_batcher.add_request(prompt, parameters or MODEL_PARAMS)
                except RuntimeError as e:
                    if "Event loop" in str(e):
                        logger.warning("Batching failed due to event loop issue, falling back to direct call")
                    else:
                        raise
        
        # Direct call for large requests or when batching disabled/failed
        client = await get_singleton_client()
        return await client.generate_response(prompt, parameters)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return f"Error: Configuration issue - {str(e)[:100]}"
    except RuntimeError as e:
        if "Event loop" in str(e):
            logger.error("Event loop issue in call_huggingface")
            await reset_singleton_client()
            return "Error: Event loop issue. Please try again."
        raise
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
        # For now, fall back to non-streaming due to event loop issues
        result = await client.generate_response(prompt, base_params)
        yield result
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