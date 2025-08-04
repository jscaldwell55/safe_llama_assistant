import requests
import json
import logging
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
import hashlib
import time
from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """Client for interacting with Hugging Face Inference Endpoints"""
    
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        self.token = token
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        })
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0)
        self.session.mount('https://', adapter)
        logger.info(f"HuggingFaceClient initialized with endpoint: {endpoint[:30]}...")
    
    def _cache_key(self, prompt: str, parameters: Dict[str, Any]) -> str:
        cache_data = json.dumps({"prompt": prompt, "params": parameters}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()

    @lru_cache(maxsize=100)
    def _cached_generate(self, cache_key: str, prompt: str, params_json: str) -> str:
        parameters = json.loads(params_json)
        return self._generate_response_internal(prompt, parameters)

    def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        cache_key = self._cache_key(prompt, parameters)
        params_json = json.dumps(parameters, sort_keys=True)
        return self._cached_generate(cache_key, prompt, params_json)

    def _generate_response_internal(self, prompt: str, parameters: Dict[str, Any]) -> str:
        payload = {"inputs": prompt, "parameters": parameters}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to HF endpoint (attempt {attempt + 1}/{max_retries})")
                response = self.session.post(self.endpoint, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                    generated_text = result[0]["generated_text"]
                    # The response often includes the prompt, so we strip it.
                    if generated_text.startswith(prompt):
                        return generated_text[len(prompt):].strip()
                    return generated_text
                else:
                    logger.warning(f"Unexpected response format: {result}")
                    return str(result)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "Error: Could not connect to the model service."
        return "Error: Exceeded max retries for model service."

# --- Convenience Functions that use the lazy-loaded client ---

def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to call the model via the lazy-loaded client."""
    client = get_hf_client()
    return client.generate_response(prompt, parameters)

def call_base_assistant(prompt: str) -> str:
    """Calls the model with parameters optimized for conversational responses."""
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1,
        "max_new_tokens": 300, "stop_sequences": ["User:", "Human:", "\n\n"]
    })
    return call_huggingface(prompt, base_params)

def call_answerability_agent(prompt: str) -> Tuple[str, str]:
    """Calls the model with parameters optimized for structured JSON classification."""
    params = {
        "temperature": 0.01, "max_new_tokens": 60, "top_p": 0.1, "stop_sequences": ["}"]
    }
    raw_response = call_huggingface(prompt, params)
    
    # Clean up response to ensure it's valid JSON
    if "```json" in raw_response:
        cleaned_response = raw_response.split("```json")[1].split("```")[0].strip()
    else:
        cleaned_response = raw_response
    
    if not cleaned_response.endswith('}'):
        cleaned_response += '}'
        
    try:
        data = json.loads(cleaned_response)
        classification = data.get("classification", "NOT_ANSWERABLE")
        reason = data.get("reason", "Could not determine answerability from response.")
        return classification, reason
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from answerability agent. Response: '{cleaned_response}'")
        return "NOT_ANSWERABLE", "Invalid format from classification model."

# --- LAZY-LOADING FUNCTION ---
# This is the function that was missing from your file.
_hf_client_instance = None
def get_hf_client():
    """Lazy-loads and returns a single instance of the HuggingFaceClient."""
    global _hf_client_instance
    if _hf_client_instance is None:
        _hf_client_instance = HuggingFaceClient()
    return _hf_client_instance
