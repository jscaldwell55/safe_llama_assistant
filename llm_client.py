import requests
import json
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import hashlib
from config import HF_TOKEN, HF_INFERENCE_ENDPOINT, MODEL_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """Client for interacting with Hugging Face Inference Endpoints"""
    
    def __init__(self, token: str = HF_TOKEN, endpoint: str = HF_INFERENCE_ENDPOINT):
        self.token = token
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        # Initialize session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0  # We handle retries manually
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        # Don't perform any network operations during init
        logger.info(f"HuggingFaceClient initialized with endpoint: {endpoint[:30]}...")
    
    def _cache_key(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate a cache key from prompt and parameters"""
        cache_data = json.dumps({"prompt": prompt, "params": parameters}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    @lru_cache(maxsize=100)
    def _cached_generate(self, cache_key: str, prompt: str, params_json: str) -> str:
        """Cached version of generate_response. Uses cache_key for caching."""
        parameters = json.loads(params_json)
        return self._generate_response_internal(prompt, parameters)
    
    def generate_response(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response from the Hugging Face model with caching.
        
        Args:
            prompt (str): The input prompt to send to the model
            parameters (dict, optional): Model generation parameters
            
        Returns:
            str: The generated response text
        """
        if parameters is None:
            parameters = MODEL_PARAMS.copy()
        
        # Generate cache key and call cached version
        cache_key = self._cache_key(prompt, parameters)
        params_json = json.dumps(parameters, sort_keys=True)
        
        logger.info(f"Cache lookup for key: {cache_key[:8]}...")
        return self._cached_generate(cache_key, prompt, params_json)
    
    def _generate_response_internal(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Internal method that actually makes the API call"""
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }
        
        # Retry logic for service unavailable errors
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to HF endpoint (attempt {attempt + 1}/{max_retries})")
                response = self.session.post(
                    self.endpoint, 
                    json=payload,
                    timeout=60  # Increased timeout for longer responses
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Handle different response formats from HF Inference Endpoints
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        generated_text = result[0]["generated_text"]
                        # Remove the original prompt from the response if it's included
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].strip()
                        return generated_text
                    else:
                        return str(result[0])
                elif isinstance(result, dict) and "generated_text" in result:
                    generated_text = result["generated_text"]
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    return generated_text
                else:
                    logger.warning(f"Unexpected response format: {result}")
                    return str(result)
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    logger.warning(f"Service unavailable (503), retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"HTTP error after {attempt + 1} attempts: {e}")
                    return f"I apologize, but I'm having trouble connecting to my services right now. Please try again in a moment."
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed, retrying in {retry_delay} seconds: {e}")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    return f"I'm sorry, I'm experiencing connection issues. Please try again later."
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response: {e}")
                return "I received an unexpected response format. Please try again."
            except Exception as e:
                logger.error(f"Unexpected error during model inference: {e}")
                return f"I encountered an unexpected error. Please try again or start a new conversation."
    
    def health_check(self) -> bool:
        """
        Check if the Hugging Face endpoint is available.
        
        Returns:
            bool: True if endpoint is healthy, False otherwise
        """
        try:
            test_payload = {
                "inputs": "Hello",
                "parameters": {"max_new_tokens": 1}
            }
            response = self.session.post(
                self.endpoint,
                json=test_payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Global client instance
hf_client = HuggingFaceClient()

def call_huggingface(prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function for calling the Hugging Face model.
    
    Args:
        prompt (str): The input prompt
        parameters (dict, optional): Model generation parameters
        
    Returns:
        str: The generated response
    """
    return hf_client.generate_response(prompt, parameters)

def call_base_assistant(prompt: str) -> str:
    """
    Call the base assistant with parameters optimized for natural conversation.
    
    Args:
        prompt (str): The formatted prompt for the base assistant
        
    Returns:
        str: The assistant's response
    """
    base_params = MODEL_PARAMS.copy()
    base_params.update({
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 150,  # Reduced to prevent over-generation
        "stop_sequences": [
            "User:", "Human:", "Assistant:", "\n\n\n",
            "```", "def ", "import ", "from ",  # Stop code generation
            "(Note:", "Note:", "Please go ahead"  # Stop inappropriate notes
        ]
    })
    
    response = hf_client.generate_response(prompt, base_params)
    
    # Clean up response - remove any accidental dialogue generation
    if "User:" in response or "Human:" in response:
        response = response.split("User:")[0].split("Human:")[0].strip()
    
    # Remove code blocks and programming content
    if "```" in response:
        # Remove everything from first ``` onwards
        response = response.split("```")[0].strip()
    
    # Remove any lines that look like code
    lines = response.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip lines that look like code
        if (line.strip().startswith('def ') or 
            line.strip().startswith('import ') or 
            line.strip().startswith('from ') or
            line.strip().startswith('print(') or
            'def greet' in line or
            '= f"Hello' in line):
            continue
        filtered_lines.append(line)
    
    response = '\n'.join(filtered_lines).strip()
    
    # Remove inappropriate notes
    if "(Note:" in response:
        response = response.split("(Note:")[0].strip()
    
    return response

def call_guard_agent(prompt: str) -> str:
    """
    Call the guard agent using the same base model with optimized parameters.
    
    Args:
        prompt (str): The formatted prompt for the guard agent
        
    Returns:
        str: The guard agent's verdict
    """
    guard_params = MODEL_PARAMS.copy()
    guard_params.update({
        "temperature": 0.1,  # Very low temperature for consistent evaluation
        "max_new_tokens": 150,  # Increased to allow for intent and reasoning
        "top_p": 0.7,
        "repetition_penalty": 1.0  # No penalty for guard - we want consistent format
    })
    return hf_client.generate_response(prompt, guard_params)