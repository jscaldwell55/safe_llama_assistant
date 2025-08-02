import json
import logging
from typing import Dict, Any, Tuple
from llm_client import call_guard_agent
from prompt import format_guard_prompt
from config import DEFAULT_FALLBACK_MESSAGE, ENABLE_GUARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuardAgent:
    """
    LLM-based guard agent for evaluating assistant responses.
    """
    
    def __init__(self):
        self.enabled = ENABLE_GUARD
        self.fallback_message = DEFAULT_FALLBACK_MESSAGE
    
    def evaluate_response(
        self, 
        context: str, 
        user_question: str, 
        assistant_response: str
    ) -> Tuple[bool, str, str]:
        """
        Evaluate whether an assistant response is safe and appropriate.
        
        Args:
            context (str): The retrieved context from RAG
            user_question (str): The user's original question
            assistant_response (str): The assistant's draft response
            
        Returns:
            Tuple[bool, str, str]: (is_approved, final_response, guard_reasoning)
        """
        if not self.enabled:
            return True, assistant_response, "Guard disabled"
        
        try:
            # Format prompt for guard agent
            guard_prompt = format_guard_prompt(context, user_question, assistant_response)
            
            # Get guard evaluation
            guard_output = call_guard_agent(guard_prompt)
            
            # Parse guard response
            verdict, reasoning = self._parse_guard_response(guard_output)
            
            if verdict == "APPROVE":
                logger.info(f"Guard approved response for query: {user_question[:50]}...")
                return True, assistant_response, reasoning
            else:
                logger.warning(f"Guard rejected response: {reasoning}")
                return False, self.fallback_message, reasoning
                
        except Exception as e:
            logger.error(f"Guard evaluation failed: {e}")
            # Fail safe - return fallback message if guard fails
            return False, self.fallback_message, f"Guard evaluation error: {str(e)}"
    
    def _parse_guard_response(self, guard_output: str) -> Tuple[str, str]:
        """
        Parse the guard agent's response to extract verdict and reasoning.
        
        Args:
            guard_output (str): Raw output from guard agent
            
        Returns:
            Tuple[str, str]: (verdict, reasoning)
        """
        try:
            guard_output_clean = guard_output.strip()
            
            # Check if this is an error response from the LLM client
            if guard_output_clean.startswith("[Error:"):
                logger.error(f"Guard agent received error response: {guard_output_clean}")
                return "REJECT", f"Guard service unavailable: {guard_output_clean}"
            
            guard_output_upper = guard_output_clean.upper()
            
            # Simple keyword-based parsing for APPROVE/REJECT
            if "APPROVE" in guard_output_upper:
                return "APPROVE", "Response approved by guard"
            elif "REJECT" in guard_output_upper:
                return "REJECT", "Response rejected by guard"
            else:
                # If unclear, err on the side of caution
                logger.warning(f"Unclear guard response: {guard_output}")
                return "REJECT", "Unclear guard response - defaulting to reject"
                
        except Exception as e:
            logger.error(f"Error parsing guard response: {e}")
            return "REJECT", f"Guard parsing error: {str(e)}"
    
    def quick_safety_check(self, text: str) -> bool:
        """
        Perform a basic length check.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text passes basic checks
        """
        # Only check for extremely short or empty responses
        if len(text.strip()) < 10:
            logger.warning(f"Response too short: {len(text.strip())} characters")
            return False
        
        return True

# Global guard agent instance
guard_agent = GuardAgent()

def evaluate_response(
    context: str, 
    user_question: str, 
    assistant_response: str
) -> Tuple[bool, str, str]:
    """
    Convenience function for evaluating assistant responses.
    
    Args:
        context (str): The retrieved context from RAG
        user_question (str): The user's original question
        assistant_response (str): The assistant's draft response
        
    Returns:
        Tuple[bool, str, str]: (is_approved, final_response, guard_reasoning)
    """
    return guard_agent.evaluate_response(context, user_question, assistant_response)

# Legacy functions for backward compatibility
def is_safe_response(text: str) -> bool:
    """Legacy function - returns basic safety check"""
    return guard_agent.quick_safety_check(text)

def redirect_response() -> str:
    """Legacy function - returns fallback message"""
    return DEFAULT_FALLBACK_MESSAGE
