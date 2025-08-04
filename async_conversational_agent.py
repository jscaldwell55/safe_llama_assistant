import logging
from typing import Optional, Dict, Any, List
import asyncio
from async_llm_client import AsyncHuggingFaceClient
from async_rag import AsyncRAGSystem
from guard import GuardAgent
from prompt import PromptTemplates
from context_formatter import ContextFormatter
from config import (
    ENABLE_GUARD,
    MAX_CONVERSATION_TURNS,
    MAX_CONTEXT_LENGTH
)

logger = logging.getLogger(__name__)

class AsyncConversationalAgent:
    """
    Async conversational agent that combines RAG, LLM, and guard capabilities.
    """
    
    def __init__(self):
        self.llm_client = None
        self.rag_system = None
        self.guard = GuardAgent() if ENABLE_GUARD else None
        self.formatter = ContextFormatter()
        self.conversation_history = []
        self.turn_count = 0
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize all async components."""
        if self.is_initialized:
            return
        
        # Initialize RAG system
        self.rag_system = AsyncRAGSystem()
        await self.rag_system.initialize()
        
        # LLM client will be initialized in context manager
        self.is_initialized = True
        logger.info("AsyncConversationalAgent initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        self.llm_client = AsyncHuggingFaceClient()
        await self.llm_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.llm_client:
            await self.llm_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.rag_system:
            await self.rag_system.close()
    
    def _manage_conversation_history(self):
        """Keep conversation history within reasonable bounds."""
        # Keep only last 5 exchanges if history gets too long
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Check total context length
        total_length = sum(len(entry.get("user", "")) + len(entry.get("assistant", "")) 
                          for entry in self.conversation_history)
        
        # If too long, keep only recent history
        while total_length > MAX_CONTEXT_LENGTH and len(self.conversation_history) > 1:
            self.conversation_history.pop(0)
            total_length = sum(len(entry.get("user", "")) + len(entry.get("assistant", "")) 
                             for entry in self.conversation_history)
    
    async def process_query(self, user_query: str) -> str:
        """
        Process a user query asynchronously through the full pipeline.
        
        Args:
            user_query: The user's input question
            
        Returns:
            The assistant's response
        """
        try:
            # Check conversation limits
            if self.turn_count >= MAX_CONVERSATION_TURNS:
                return "We've reached the conversation limit. Please start a new session to continue."
            
            # Safety check (if guard is enabled)
            if self.guard:
                is_safe, safety_response = self.guard.check_input_safety(user_query)
                if not is_safe:
                    logger.warning(f"Unsafe query detected: {user_query}")
                    return safety_response
            
            # Retrieve relevant context asynchronously
            logger.info(f"Processing query: {user_query[:100]}...")
            retrieved_docs = await self.rag_system.retrieve(user_query)
            
            # Format context
            context = self.formatter.format_retrieved_context(retrieved_docs)
            
            # Build conversation prompt
            conversation_context = self.formatter.format_conversation_history(self.conversation_history)
            
            # Create full prompt
            if context:
                prompt = PromptTemplates.build_conversational_prompt(
                    query=user_query,
                    context=context,
                    conversation_history=conversation_context
                )
            else:
                prompt = PromptTemplates.build_no_context_prompt(
                    query=user_query,
                    conversation_history=conversation_context
                )
            
            # Generate response asynchronously
            response = await self.llm_client.generate_response(prompt)
            
            # Post-process response
            if response.startswith("Error:"):
                logger.error(f"LLM error: {response}")
                return "I encountered an error processing your request. Please try rephrasing your question."
            
            # Validate response (if guard is enabled)
            if self.guard and context:
                is_grounded = self.guard.validate_response_grounding(response, context)
                if not is_grounded:
                    logger.warning("Response not well grounded, using fallback")
                    response = "I can only provide information based on our documentation. Could you please rephrase your question?"
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_query,
                "assistant": response
            })
            self._manage_conversation_history()
            self.turn_count += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I encountered an error processing your request. Please try again."
    
    async def process_batch_queries(self, queries: List[str]) -> List[str]:
        """
        Process multiple queries concurrently.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of responses
        """
        tasks = [self.process_query(query) for query in queries]
        return await asyncio.gather(*tasks)
    
    def reset_conversation(self):
        """Reset the conversation state."""
        self.conversation_history = []
        self.turn_count = 0
        logger.info("Conversation reset")
    
    async def rebuild_index(self):
        """Rebuild the FAISS index from PDFs asynchronously."""
        logger.info("Rebuilding index...")
        await self.rag_system.build_index()
        logger.info("Index rebuilt successfully")