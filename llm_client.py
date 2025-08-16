# llm_client.py - Clean Working Version with Event Loop Fix

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
RETRY_DELAY_BASE = 1.0
RETRY_DELAY_MAX = 10.0

# ============================================================================
# EVENT LOOP MANAGEMENT
# ============================================================================

def get_or_create_event_loop():
    """Get the current event loop or create a new one if needed"""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# ============================================================================
# OUTPUT CLEANING
# ============================================================================

def clean_model_output(text: str) -> str:
    """Clean and format model output to natural language"""
    if not text:
        return ""
    
    original_text = text
    
    # Remove extraction format artifacts
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
            if not line:
                continue
            # Remove bullet markers
            line = re.sub(r'^[•\-\*]\s*', '', line)
            # Skip headers
            if line and not line.endswith(':'):
                # Ensure line ends with period
                if line and line[-1] not in '.!?':
                    line += '.'
                cleaned_lines.append(line)
        
        # Join facts with better flow
        if cleaned_lines:
            text = ' '.join(cleaned_lines)
    
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
        r"^\[.*?\]\s*",
        r"^Note:.*?\n",
    ]
    
    for pattern in cot_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up formatting artifacts
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'__(.*?)__', r'\1', text)  # Remove underline
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*', r'\1', text)  # Remove italics
    
    # Fix spacing
    text = re.sub(r'\n\s*\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Ensure complete sentence
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text.strip()

# ============================================================================
# HUGGINGFACE CLIENT
# ============================================================================

class HuggingFaceClient:
    """Client for HuggingFace inference endpoint"""
    
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        if not token:
            raise ValueError("HF_TOKEN is not configured")
        if not endpoint:
            raise ValueError("HF_INFERENCE_ENDPOINT is not configured")
        
        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        self.session = None
        self.session_lock = asyncio.Lock()
        self._closed = False
        self._loop = None
        
        self.request_count = 0
        self.error_count = 0
        
        logger.info(f"HuggingFaceClient initialized: {self.endpoint[:60]}...")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session with current event loop"""
        if self._closed:
            raise RuntimeError("Client is closed")
        
        current_loop = get_or_create_event_loop()
        
        async with self.session_lock:
            # Check if we need a new session
            if (self.session is None or 
                self.session.closed or 
                self._loop != current_loop or
                (self._loop and self._loop.is_closed())):
                
                # Close old session if exists
                if self.session and not self.session.closed:
                    try:
                        await self.session.close()
                    except Exception as e:
                        logger.warning(f"Error closing old session: {e}")
                
                # Create new session
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                    force_close=False,
                    keepalive_timeout=30
                )
                
                self.session = aiohttp.ClientSession(
                    headers=self.headers,
                    connector=connector
                )
                self._loop = current_loop
                logger.info("Created new aiohttp session")
            
            return self.session
    
    async def close(self):
        """Close the session"""
        self._closed = True
        async with self.session_lock:
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
                self.session = None
                self._loop = None
    
    async def reset(self):
        """Reset the client"""
        logger.info("Resetting HuggingFaceClient")
        await self.close()
        self._closed = False
        self._loop = None
        self.session = None
    
    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate response from the model"""
        if self._closed:
            await self.reset()
        
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        start_time = time.time()
        self.request_count += 1
        
        timeout = aiohttp.ClientTimeout(total=30, sock_read=25)
        
        try:
            session = await self._get_session()
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            self.error_count += 1
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
                
                if status == 503:
                    logger.info("Model loading, waiting...")
                    await asyncio.sleep(2)
                    return await self.generate_response(prompt, parameters)
                
                text = await response.text()
                
                if status != 200:
                    logger.error(f"API returned {status}: {text[:200]}")
                    self.error_count += 1
                    
                    if status == 429:
                        return "Error: Rate limit exceeded"
                    elif status == 500:
                        return "Error: Server error"
                    else:
                        return f"Error: Status {status}"
                
                # Parse response
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    cleaned = clean_model_output(text)
                    if cleaned:
                        return cleaned
                    return "Error: Invalid response format"
                
                # Extract text
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
                    return "Error: No response generated"
                
                # Remove prompt from response
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].lstrip()
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                if elapsed_ms > 5000:
                    logger.warning(f"[PERF] Slow generation: {elapsed_ms}ms")
                else:
                    logger.info(f"[PERF] Generation completed in {elapsed_ms}ms")
                
                return clean_model_output(generated)
                
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            self.error_count += 1
            return "Error: Request timed out"
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}")
            self.error_count += 1
            if "Cannot connect" in str(e) or "Connection reset" in str(e):
                await self.reset()
            return f"Error: Connection failed"
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.error("Event loop closed, resetting")
                await self.reset()
                return "Error: Event loop issue. Please try again."
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self.error_count += 1
            return "Error: An unexpected error occurred"

# ============================================================================
# REQUEST BATCHING (OPTIONAL)
# ============================================================================

@dataclass
class BatchRequest:
    prompt: str
    parameters: Dict[str, Any]
    future: asyncio.Future
    timestamp: float
    retry_count: int = 0

class RequestBatcher:
    """Batch requests for efficiency (optional)"""
    
    def __init__(self, max_batch_size: int = 4, timeout_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests: deque = deque()
        self.batch_lock = asyncio.Lock()
        self.batch_task = None
        self._closed = False
    
    async def add_request(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Add request to batch"""
        if self._closed:
            raise RuntimeError("Batcher is closed")
        
        loop = get_or_create_event_loop()
        future = loop.create_future()
        request = BatchRequest(prompt, parameters, future, time.time())
        
        async with self.batch_lock:
            self.pending_requests.append(request)
            
            if self.batch_task is None or self.batch_task.done():
                self.batch_task = asyncio.create_task(self._process_batch())
            
            if len(self.pending_requests) >= self.max_batch_size:
                if self.batch_task and not self.batch_task.done():
                    self.batch_task.cancel()
                self.batch_task = asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process pending batch"""
        if self._closed:
            return
        
        await asyncio.sleep(self.timeout_ms / 1000)
        
        async with self.batch_lock:
            if not self.pending_requests:
                return
            
            batch = []
            for _ in range(min(self.max_batch_size, len(self.pending_requests))):
                batch.append(self.pending_requests.popleft())
            
            if not batch:
                return
        
        logger.info(f"[BATCH] Processing {len(batch)} requests")
        
        tasks = [self._process_single_with_retry(req) for req in batch]
        
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
        """Process single request with retry"""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                client = await get_singleton_client()
                result = await client.generate_response(request.prompt, request.parameters)
                
                if result.startswith("Error:") and "Event loop" not in result and attempt < MAX_RETRIES - 1:
                    raise Exception(result)
                
                return result
                
            except Exception as e:
                last_error = e
                request.retry_count = attempt + 1
                
                if "Event loop" in str(e) or "Session is closed" in str(e):
                    logger.warning("Session issue detected, resetting")
                    await reset_singleton_client()
                
                if attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_DELAY_BASE * (2 ** attempt), RETRY_DELAY_MAX)
                    logger.warning(f"Request retry {attempt + 1}/{MAX_RETRIES} after {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {MAX_RETRIES} attempts")
        
        return f"Error: Failed after {MAX_RETRIES} attempts"
    
    async def close(self):
        """Close the batcher"""
        self._closed = True
        
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()
        
        async with self.batch_lock:
            while self.pending_requests:
                req = self.pending_requests.popleft()
                if not req.future.done():
                    req.future.cancel()

# Global batcher
request_batcher = RequestBatcher() if ENABLE_REQUEST_BATCHING else None

# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_client_instance: Optional[HuggingFaceClient] = None
_client_lock = asyncio.Lock()

async def get_singleton_client() -> HuggingFaceClient:
    """Get singleton client"""
    global _client_instance
    
    async with _client_lock:
        if _client_instance is None or _client_instance._closed:
            _client_instance = HuggingFaceClient()
        return _client_instance

async def reset_singleton_client():
    """Reset singleton client"""
    global _client_instance
    
    async with _client_lock:
        if _client_instance:
            await _client_instance.reset()

# ============================================================================
# CLEANUP
# ============================================================================

async def cleanup():
    """Cleanup resources"""
    global _client_instance, request_batcher
    
    logger.info("Starting cleanup...")
    
    if request_batcher:
        try:
            await request_batcher.close()
            logger.info("Closed batcher")
        except Exception as e:
            logger.warning(f"Error closing batcher: {e}")
    
    if _client_instance:
        try:
            await _client_instance.close()
            _client_instance = None
            logger.info("Closed client")
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
    
    logger.info("Cleanup completed")

# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

async def call_huggingface_with_retry(
    prompt: str,
    parameters: Optional[Dict[str, Any]] = None,
    max_retries: int = MAX_RETRIES
) -> str:
    """Call HuggingFace with retry logic"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = await call_huggingface(prompt, parameters)
            
            if result.startswith("Error:") and "Event loop" in result and attempt < max_retries - 1:
                logger.warning("Event loop error, resetting")
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
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s")
                await asyncio.sleep(delay)
    
    return f"Error: Failed after {max_retries} attempts"

async def call_huggingface(
    prompt: str,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """Main entry point for HuggingFace calls"""
    try:
        # Try batching for small requests
        if request_batcher and ENABLE_REQUEST_BATCHING and not request_batcher._closed:
            if parameters is None or parameters.get("max_new_tokens", 0) <= 100:
                try:
                    return await request_batcher.add_request(prompt, parameters or MODEL_PARAMS)
                except RuntimeError as e:
                    if "Event loop" in str(e):
                        logger.warning("Batching failed, using direct call")
                    else:
                        raise
        
        # Direct call
        client = await get_singleton_client()
        return await client.generate_response(prompt, parameters)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return f"Error: Configuration issue"
    except RuntimeError as e:
        if "Event loop" in str(e):
            logger.error("Event loop issue")
            await reset_singleton_client()
            return "Error: Event loop issue. Please try again."
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return "Error: An unexpected error occurred"

async def call_base_assistant(prompt: str) -> str:
    """Call base assistant"""
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 200,
        "stop": ["\nUser:", "\nHuman:", "\nAssistant:", "###"]
    })
    return await call_huggingface_with_retry(prompt, base_params)

async def call_guard_agent(prompt: str) -> str:
    """Call guard agent"""
    from config import GUARD_MODEL_PARAMS
    return await call_huggingface(prompt, GUARD_MODEL_PARAMS)

async def call_intent_classifier(prompt: str) -> str:
    """Call intent classifier"""
    from config import INTENT_CLASSIFIER_PARAMS
    return await call_huggingface(prompt, INTENT_CLASSIFIER_PARAMS)

async def call_empathetic_companion(prompt: str) -> str:
    """Call empathetic companion"""
    from config import EMPATHETIC_COMPANION_PARAMS
    return await call_huggingface_with_retry(prompt, EMPATHETIC_COMPANION_PARAMS)

async def call_information_navigator(prompt: str) -> str:
    """Call information navigator"""
    from config import INFORMATION_NAVIGATOR_PARAMS
    return await call_huggingface_with_retry(prompt, INFORMATION_NAVIGATOR_PARAMS)

async def call_bridge_synthesizer(prompt: str) -> str:
    """Call bridge synthesizer"""
    from config import BRIDGE_SYNTHESIZER_PARAMS
    return await call_huggingface_with_retry(prompt, BRIDGE_SYNTHESIZER_PARAMS)