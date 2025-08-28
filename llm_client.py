# llm_client.py - Claude 3.5 Sonnet Integration (Non-streaming)

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from anthropic import AsyncAnthropic
import anthropic

from config import (
    ANTHROPIC_API_KEY, 
    CLAUDE_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    NO_CONTEXT_FALLBACK_MESSAGE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPT - Core Safety Through Design
# ============================================================================

SYSTEM_PROMPT = """You are a pharmaceutical information assistant for Journvax. Your responses must follow these critical rules:

1. **ONLY use information from the provided context** - Never use external knowledge
2. **If no context is provided or the context is empty**, respond exactly with: "I'm sorry, I don't have any information on that. Can I assist you with something else?"
3. **Never generate creative content** like stories, poems, or fictional scenarios
4. **Never provide personal medical advice** - Only share factual information from the documentation
5. **Always stay focused on Journvax** - Don't discuss unrelated topics
6. **Be concise and factual** - Present information as direct statements from the documentation

Remember: You can ONLY discuss what is explicitly stated in the context provided. If information isn't in the context, you must use the fallback message."""

# ============================================================================
# CLAUDE CLIENT
# ============================================================================

class ClaudeClient:
    """Client for Anthropic's Claude API"""
    
    def __init__(self, api_key: str = ANTHROPIC_API_KEY):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = CLAUDE_MODEL
        self.request_count = 0
        self.error_count = 0
        
        logger.info(f"ClaudeClient initialized with model: {self.model}")
    
    async def generate_response(
        self, 
        user_query: str, 
        context: str = "",
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Generate response using Claude with strict context grounding"""
        
        start_time = time.time()
        self.request_count += 1
        
        # Build the user message with context
        if context and len(context.strip()) > 50:
            user_message = f"""Context from Journvax documentation:
{context}

User Question: {user_query}

Please answer using ONLY the information provided in the context above."""
        else:
            # No context - should trigger fallback
            user_message = f"""No documentation context available.

User Question: {user_query}

Since no context is provided, please respond with the appropriate fallback message."""
        
        # Build messages list
        messages = []
        
        # Add conversation history if provided (limit to recent messages)
        if conversation_history:
            # Take last 10 messages to stay within context limits
            recent_history = conversation_history[-10:]
            for msg in recent_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            logger.info(f"Sending request to Claude. Context length: {len(context)} chars")
            
            response = await self.client.messages.create(
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            generated_text = response.content[0].text if response.content else ""
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            if elapsed_ms > 3000:
                logger.warning(f"[PERF] Slow Claude response: {elapsed_ms}ms")
            else:
                logger.info(f"[PERF] Claude response in {elapsed_ms}ms")
            
            if not generated_text:
                logger.warning("Empty response from Claude")
                return NO_CONTEXT_FALLBACK_MESSAGE
            
            # Clean up any potential formatting issues
            generated_text = generated_text.strip()
            
            logger.debug(f"Generated response length: {len(generated_text)} chars")
            
            return generated_text
            
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            self.error_count += 1
            return "I'm experiencing high demand. Please try again in a moment."
            
        except anthropic.APIError as e:
            logger.error(f"API error: {e}")
            self.error_count += 1
            return NO_CONTEXT_FALLBACK_MESSAGE
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self.error_count += 1
            return NO_CONTEXT_FALLBACK_MESSAGE
    
    async def close(self):
        """Close the client"""
        # AsyncAnthropic handles cleanup internally
        logger.info("ClaudeClient closed")

# ============================================================================
# SINGLETON MANAGEMENT
# ============================================================================

_client_instance: Optional[ClaudeClient] = None

async def get_singleton_client() -> ClaudeClient:
    """Get singleton client"""
    global _client_instance
    
    if _client_instance is None:
        _client_instance = ClaudeClient()
    return _client_instance

# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

async def call_claude(
    user_query: str,
    context: str = "",
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """Main entry point for Claude calls"""
    try:
        client = await get_singleton_client()
        return await client.generate_response(user_query, context, conversation_history)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return NO_CONTEXT_FALLBACK_MESSAGE
    except Exception as e:
        logger.error(f"Unexpected error in call_claude: {e}", exc_info=True)
        return NO_CONTEXT_FALLBACK_MESSAGE

# ============================================================================
# CLEANUP
# ============================================================================

async def cleanup():
    """Cleanup resources"""
    global _client_instance
    
    logger.info("Starting Claude client cleanup...")
    
    if _client_instance:
        try:
            await _client_instance.close()
            _client_instance = None
            logger.info("Claude client closed")
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
    
    logger.info("Cleanup completed")