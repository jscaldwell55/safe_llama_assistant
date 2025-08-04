import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)

class ContextFormatter:
    """Simplified context formatting that trusts the model to understand relevance"""
    
    def __init__(self):
        self.max_context_length = 4000  # Character limit for context
    
    def format_enhanced_context(self, chunks: List[str], query: str, 
                              conversation_context: str = "", 
                              conversation_entities: List[str] = None) -> str:
        """
        Format context simply and clearly for the model.
        Trust the model to determine what's relevant.
        """
        
        if not chunks:
            return ""
        
        # Simple deduplication - remove exact or near-exact duplicates
        unique_chunks = self._simple_deduplicate(chunks)
        
        # Build context
        context_parts = []
        
        # Add conversation history if it exists
        if conversation_context:
            context_parts.append(conversation_context)
            context_parts.append("")  # Blank line
        
        # Add retrieved information
        context_parts.append("Retrieved information:")
        context_parts.append("")
        
        # Add chunks up to length limit
        current_length = len("\n".join(context_parts))
        
        for i, chunk in enumerate(unique_chunks):
            # Simple formatting - let the model parse naturally
            chunk_text = chunk.strip()
            
            # Check length
            if current_length + len(chunk_text) + 50 > self.max_context_length:
                # Add note about truncation
                context_parts.append("[Additional information truncated due to length]")
                break
            
            # Add chunk with simple separator
            if i > 0:
                context_parts.append("---")
            context_parts.append(chunk_text)
            current_length += len(chunk_text) + 10
        
        formatted_context = "\n".join(context_parts)
        
        logger.info(f"Formatted {len(unique_chunks)} chunks into {len(formatted_context)} chars")
        
        return formatted_context
    
    def _simple_deduplicate(self, chunks: List[str]) -> List[str]:
        """Remove obvious duplicates while preserving order"""
        seen_content = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create a normalized version for comparison
            normalized = " ".join(chunk.lower().split())[:200]  # First 200 chars normalized
            
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def extract_key_information(self, chunks: List[str], query: str) -> Dict[str, List[str]]:
        """
        Simple categorization if needed.
        But generally, we trust the model to understand context.
        """
        # For backward compatibility, return all chunks as general info
        return {
            'general_info': chunks,
            'side_effects': [],
            'dosage_info': [],
            'warnings': []
        }

# Global context formatter instance
context_formatter = ContextFormatter()