# llm_client.py - Claude 3.5 Sonnet Integration (with streaming)

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from anthropic import AsyncAnthropic
import anthropic

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    NO_CONTEXT_FALLBACK_MESSAGE # Make sure this is consistently used
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPT - Core Safety Through Design
# ============================================================================

SYSTEM_PROMPT = """You are a pharmaceutical information assistant for Journvax. Your responses must follow these critical rules:

1. **ONLY use facts and information DERIVED from the provided context** - Do not use any external or prior knowledge.
2. **If no context is provided, the context is empty, or the context does not contain the answer**, respond EXACTLY with: "I'm sorry, I don't have any information on that. Can I assist you with something else?"
3. **Never generate creative content** like stories, poems, or fictional scenarios.
4. **Never provide personal medical advice** - Only share factual information from the documentation.
5. **Always stay focused on Journvax** - Do not discuss unrelated topics.
6. **Be concise and factual** - Present information as direct statements or clear summaries from the documentation.

Remember: You can ONLY discuss what is explicitly stated or can be directly inferred from the context provided. If information isn't in the context, you must use the specified fallback message."""

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

    async def generate_response_stream(
        self,
        user_query: str,
        context: str = "",
        conversation_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate response using Claude with strict context grounding, returning a stream."""

        start_time = time.time()
        self.request_count += 1

        # Build the user message with context
        if context and len(context.strip()) > 50:
            user_message = f"""Context from Journvax documentation:
{context}

User Question: {user_query}

Please answer using ONLY the information provided in the context above."""
        else:
            user_message = f"""No documentation context available or context is irrelevant.

User Question: {user_query}

As per your instructions, please respond with the exact fallback message for no context: "{NO_CONTEXT_FALLBACK_MESSAGE}" """

        # Build messages list
        messages = []

        if conversation_history:
            recent_history = conversation_history[-10:]
            for msg in recent_history:
                # Filter out the welcome message which is added by the conversation manager
                # and explicitly handled by the system prompt or fallback in llm_client.
                # Only include actual user/assistant turns.
                if msg.get("content") != NO_CONTEXT_FALLBACK_MESSAGE:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

        messages.append({
            "role": "user",
            "content": user_message
        })

        full_response_content = []
        try:
            logger.info(f"Sending streaming request to Claude. Context length: {len(context)} chars")

            async with self.client.messages.stream(
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            ) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        if chunk.delta.text:
                            text_chunk = chunk.delta.text
                            full_response_content.append(text_chunk)
                            yield text_chunk
                    elif chunk.type == "message_stop":
                        # Log the final usage information if available
                        if chunk.amazon_bedrock_invocation_metrics: # Anthropic Bedrock specific
                            logger.info(f"Invocation metrics: {chunk.amazon_bedrock_invocation_metrics}")
                        elif chunk.usage: # Anthropic API specific
                            logger.info(f"Usage: {chunk.usage}")

            elapsed_ms = int((time.time() - start_time) * 1000)
            final_text = "".join(full_response_content).strip()

            if elapsed_ms > LOG_SLOW_REQUESTS_THRESHOLD_MS: # Using LOG_SLOW_REQUESTS_THRESHOLD_MS from config
                logger.warning(f"[PERF] Slow Claude streaming response (total): {elapsed_ms}ms")
            else:
                logger.info(f"[PERF] Claude streaming response (total) in {elapsed_ms}ms")

            if not final_text:
                logger.warning("Empty response from Claude stream")
                # If the stream yields nothing, ensure a fallback is provided.
                # This fallback will be validated by the orchestrator/guard.
                yield NO_CONTEXT_FALLBACK_MESSAGE
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            self.error_count += 1
            yield "I'm experiencing high demand. Please try again in a moment."
        except anthropic.APIError as e:
            logger.error(f"API error: {e}")
            self.error_count += 1
            yield NO_CONTEXT_FALLBACK_MESSAGE
        except Exception as e:
            logger.error(f"Unexpected error in Claude stream: {e}", exc_info=True)
            self.error_count += 1
            yield NO_CONTEXT_FALLBACK_MESSAGE

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
) -> AsyncGenerator[str, None]: # This function now returns an async generator
    """Main entry point for Claude calls, returning a streaming response."""
    try:
        client = await get_singleton_client()
        return client.generate_response_stream(user_query, context, conversation_history)

    except ValueError as e:
        logger.error(f"Configuration error in call_claude: {e}")
        yield NO_CONTEXT_FALLBACK_MESSAGE
    except Exception as e:
        logger.error(f"Unexpected error in call_claude streaming: {e}", exc_info=True)
        yield NO_CONTEXT_FALLBACK_MESSAGE

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