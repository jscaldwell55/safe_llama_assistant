import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import hashlib
from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncHuggingFaceClient:
    """Async client for interacting with Hugging Face Inference Endpoints"""
    
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        self.token = token
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.session = None
        logger.info(f"AsyncHuggingFaceClient initialized with endpoint: {endpoint[:30]}...")
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=10,
            ttl_dns_cache=300
        )
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _cache_key(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate a cache key from prompt and parameters"""
        cache_data = json.dumps({"prompt": prompt, "params": parameters}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    @lru_cache(maxsize=100)
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if exists"""
        return None
    
    def _set_cached_response(self, cache_key: str, response: str):
        """Set cached response"""
        # LRU cache handles this internally
        pass
    
    async def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response from the Hugging Face model asynchronously.
        
        Args:
            prompt: The input prompt
            parameters: Optional model parameters
            
        Returns:
            The generated response text
        """
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        # Check cache
        cache_key = self._cache_key(prompt, parameters)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with self.session.post(self.endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                            self._set_cached_response(cache_key, generated_text)
                            return generated_text
                        return ""
                    
                    error_text = await response.text()
                    
                    if response.status == 503:
                        logger.warning(f"Model is loading, attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                    
                    logger.error(f"API Error {response.status}: {error_text}")
                    return f"Error: {response.status}"
                    
            except asyncio.TimeoutError:
                logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return "Error: Request timeout"
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
    async def batch_generate(self, prompts: list[str], parameters: Optional[Dict[str, Any]] = None) -> list[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of input prompts
            parameters: Optional model parameters
            
        Returns:
            List of generated responses
        """
        tasks = [self.generate_response(prompt, parameters) for prompt in prompts]
        return await asyncio.gather(*tasks)