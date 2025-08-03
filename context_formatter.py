import logging
from typing import List, Dict, Tuple
from collections import Counter
import re

logger = logging.getLogger(__name__)

class ContextFormatter:
    """Enhanced context formatting with relevance scoring and deduplication"""
    
    def __init__(self):
        self.min_chunk_relevance = 0.3
        self.max_context_length = 4000  # Character limit for context
    
    def clean_chunk_text(self, chunk: str) -> str:
        """Clean chunk text by removing FAQ-style formatting and boilerplate"""
        if not chunk:
            return chunk
            
        # Remove common FAQ/Q&A formatting patterns
        patterns_to_remove = [
            r'^User Question:\s*',
            r'^Question:\s*',
            r'^Answer:\s*',
            r'^Response:\s*',
            r'^A:\s*',
            r'^Q:\s*',
            r'^\d+\.\s*Question:\s*',
            r'^\d+\.\s*Answer:\s*',
        ]
        
        cleaned = chunk
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Normalize spaces
        cleaned = cleaned.strip()
        
        return cleaned
        
    def score_chunk_relevance(self, chunk: str, query: str, conversation_entities: List[str] = None) -> float:
        """Score how relevant a chunk is to the query and conversation context"""
        if not chunk or not query:
            return 0.0
            
        chunk_lower = chunk.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Direct query term matches (weighted heavily)
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
        
        if query_words:
            word_overlap = len(query_words & chunk_words) / len(query_words)
            score += word_overlap * 0.6
        
        # Conversation entity matches
        if conversation_entities:
            entity_matches = sum(1 for entity in conversation_entities if entity.lower() in chunk_lower)
            score += min(entity_matches * 0.2, 0.3)
        
        # Medical relevance patterns
        medical_patterns = [
            r'\b(side effects?|adverse reactions?)\b',
            r'\b(dosage|dose|mg|milligrams?)\b', 
            r'\b(take|taking|taken)\b',
            r'\b(withdrawal|discontinu\w*|stopping)\b',
            r'\b(common|rare|serious)\b.*\b(effects?|reactions?)\b'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, chunk_lower) and re.search(pattern, query_lower):
                score += 0.2
                break
                
        # Penalize very short chunks
        if len(chunk.strip()) < 100:
            score *= 0.7
            
        # Bonus for chunks with complete sentences
        sentence_count = len(re.findall(r'[.!?]+', chunk))
        if sentence_count >= 2:
            score += 0.1
            
        return min(score, 1.0)
    
    def deduplicate_chunks(self, chunks: List[str], similarity_threshold: float = 0.8) -> List[str]:
        """Remove highly similar chunks"""
        if not chunks:
            return chunks
            
        deduplicated = []
        
        for chunk in chunks:
            chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
            
            is_duplicate = False
            for existing_chunk in deduplicated:
                existing_words = set(re.findall(r'\b\w+\b', existing_chunk.lower()))
                
                if chunk_words and existing_words:
                    overlap = len(chunk_words & existing_words)
                    similarity = overlap / max(len(chunk_words), len(existing_words))
                    
                    if similarity > similarity_threshold:
                        # Keep the longer, more complete chunk
                        if len(chunk) > len(existing_chunk):
                            deduplicated.remove(existing_chunk)
                            deduplicated.append(chunk)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(chunk)
                
        logger.info(f"Deduplicated {len(chunks)} chunks to {len(deduplicated)}")
        return deduplicated
    
    def format_enhanced_context(self, chunks: List[str], query: str, 
                              conversation_context: str = "", 
                              conversation_entities: List[str] = None) -> str:
        """Format context with improved structure and relevance filtering"""
        
        if not chunks:
            return ""
        
        # Clean and score chunks
        scored_chunks = []
        for chunk in chunks:
            # Clean the chunk first
            cleaned_chunk = self.clean_chunk_text(chunk)
            if not cleaned_chunk:  # Skip empty chunks after cleaning
                continue
                
            score = self.score_chunk_relevance(cleaned_chunk, query, conversation_entities)
            if score >= self.min_chunk_relevance:
                scored_chunks.append((cleaned_chunk, score))
        
        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the chunks
        relevant_chunks = [chunk for chunk, score in scored_chunks]
        
        # Deduplicate
        final_chunks = self.deduplicate_chunks(relevant_chunks)
        
        # Format the context
        context_parts = []
        
        # Add conversation context if available
        if conversation_context:
            context_parts.append("=== CONVERSATION CONTEXT ===")
            context_parts.append(conversation_context)
            context_parts.append("")
        
        # Add knowledge base context
        context_parts.append("=== KNOWLEDGE BASE INFORMATION ===")
        
        current_length = len("\n".join(context_parts))
        
        for i, chunk in enumerate(final_chunks):
            chunk_header = f"\n--- Information Source {i+1} ---"
            formatted_chunk = f"{chunk_header}\n{chunk.strip()}"
            
            # Check length limit
            if current_length + len(formatted_chunk) > self.max_context_length:
                break
                
            context_parts.append(formatted_chunk)
            current_length += len(formatted_chunk)
        
        formatted_context = "\n".join(context_parts)
        
        logger.info(f"Formatted context: {len(final_chunks)} chunks, {len(formatted_context)} chars")
        
        return formatted_context
    
    def extract_key_information(self, chunks: List[str], query: str) -> Dict[str, List[str]]:
        """Extract structured information from chunks"""
        info = {
            'side_effects': [],
            'dosage_info': [], 
            'warnings': [],
            'general_info': []
        }
        
        query_lower = query.lower()
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            
            # Categorize chunk content
            if any(term in chunk_lower for term in ['side effect', 'adverse', 'reaction']):
                info['side_effects'].append(chunk)
            elif any(term in chunk_lower for term in ['dose', 'dosage', 'mg', 'take']):
                info['dosage_info'].append(chunk)
            elif any(term in chunk_lower for term in ['warning', 'caution', 'avoid', 'serious']):
                info['warnings'].append(chunk)
            else:
                info['general_info'].append(chunk)
        
        return info

# Global context formatter instance
context_formatter = ContextFormatter()