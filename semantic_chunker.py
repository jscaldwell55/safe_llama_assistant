import nltk
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import logging

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Semantic chunking for medical documents using NLTK and langchain.
    Preserves document structure and semantic boundaries.
    """
    
    def __init__(self):
        # Initialize NLTK sentence tokenizer
        self.sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        
        # Initialize text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract document sections based on headers and structure.
        Particularly useful for FDA drug labels.
        """
        sections = []
        
        # Common FDA drug label section patterns
        section_patterns = [
            r'^\s*\d+\.?\s+([A-Z\s]+)$',  # Numbered sections
            r'^([A-Z][A-Z\s]{3,})$',      # All caps headers
            r'^([A-Z][a-z\s]+):',         # Title case with colon
            r'^\s*\*\s*([A-Z][a-z\s]+)',  # Bullet point headers
        ]
        
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    # Save previous section
                    if current_section and current_content:
                        sections.append({
                            'section': current_section,
                            'content': '\n'.join(current_content).strip(),
                            'type': 'section'
                        })
                    
                    current_section = match.group(1).strip()
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and line_stripped:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append({
                'section': current_section,
                'content': '\n'.join(current_content).strip(),
                'type': 'section'
            })
        
        # If no sections found, treat as single document
        if not sections and text.strip():
            sections.append({
                'section': 'document',
                'content': text.strip(),
                'type': 'document'
            })
        
        return sections
    
    def chunk_by_sentences(self, text: str, max_tokens: int = 600) -> List[str]:
        """
        Chunk text by sentence boundaries using NLTK.
        Preserves sentence integrity.
        """
        if not text.strip():
            return []
        
        # Use NLTK's sentence tokenizer
        try:
            sentences = self.sent_tokenizer.tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}. Falling back to regex.")
            # Fallback to simple regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed max_tokens, save current chunk
            if current_length + sentence_length > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, max_tokens: int = 600) -> List[str]:
        """
        Chunk text by paragraph boundaries.
        Preserves paragraph structure.
        """
        if not text.strip():
            return []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If paragraph alone exceeds max_tokens, split by sentences
            if paragraph_length > max_tokens:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentence_chunks = self.chunk_by_sentences(paragraph, max_tokens)
                chunks.extend(sentence_chunks)
            
            # If adding this paragraph would exceed max_tokens, save current chunk
            elif current_length + paragraph_length > max_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def semantic_chunk(self, text: str, strategy: str = "hybrid", 
                      max_tokens: int = 600) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Perform semantic chunking using specified strategy.
        
        Args:
            text: Input text to chunk
            strategy: 'sections', 'paragraphs', 'sentences', 'hybrid', 'recursive'
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        chunks_with_metadata = []
        
        if strategy == "sections":
            sections = self.extract_sections(text)
            for section in sections:
                section_chunks = self.chunk_by_paragraphs(section['content'], max_tokens)
                for i, chunk in enumerate(section_chunks):
                    metadata = {
                        'strategy': 'sections',
                        'section': section['section'],
                        'section_type': section['type'],
                        'chunk_index': i,
                        'total_chunks': len(section_chunks)
                    }
                    chunks_with_metadata.append((chunk, metadata))
        
        elif strategy == "paragraphs":
            chunks = self.chunk_by_paragraphs(text, max_tokens)
            for i, chunk in enumerate(chunks):
                metadata = {
                    'strategy': 'paragraphs',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                chunks_with_metadata.append((chunk, metadata))
        
        elif strategy == "sentences":
            chunks = self.chunk_by_sentences(text, max_tokens)
            for i, chunk in enumerate(chunks):
                metadata = {
                    'strategy': 'sentences',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                chunks_with_metadata.append((chunk, metadata))
        
        elif strategy == "recursive":
            chunks = self.recursive_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                metadata = {
                    'strategy': 'recursive',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                chunks_with_metadata.append((chunk, metadata))
        
        elif strategy == "hybrid":
            # Try sections first, fall back to paragraphs
            sections = self.extract_sections(text)
            if len(sections) > 1:
                # Document has clear sections
                for section in sections:
                    section_chunks = self.chunk_by_paragraphs(section['content'], max_tokens)
                    for i, chunk in enumerate(section_chunks):
                        metadata = {
                            'strategy': 'hybrid_sections',
                            'section': section['section'],
                            'section_type': section['type'],
                            'chunk_index': i,
                            'section_total_chunks': len(section_chunks)
                        }
                        chunks_with_metadata.append((chunk, metadata))
            else:
                # No clear sections, use paragraph chunking
                chunks = self.chunk_by_paragraphs(text, max_tokens)
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'strategy': 'hybrid_paragraphs',
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                    chunks_with_metadata.append((chunk, metadata))
        
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Add global metadata
        for i, (chunk, metadata) in enumerate(chunks_with_metadata):
            metadata.update({
                'global_chunk_id': i,
                'chunk_length': len(chunk),
                'max_tokens': max_tokens
            })
        
        return chunks_with_metadata
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using NLTK.
        Useful for drug names, organizations, etc.
        """
        try:
            # Tokenize text into sentences and words
            sentences = self.sent_tokenizer.tokenize(text)
            entities = {}
            
            for sentence in sentences:
                # Tokenize words
                tokens = nltk.word_tokenize(sentence)
                # Part-of-speech tagging
                pos_tags = nltk.pos_tag(tokens)
                # Named entity recognition
                ne_tree = nltk.ne_chunk(pos_tags, binary=False)
                
                # Extract entities from the tree
                for subtree in ne_tree:
                    if hasattr(subtree, 'label'):
                        entity_type = subtree.label()
                        entity_text = ' '.join(word for word, tag in subtree)
                        
                        if entity_type not in entities:
                            entities[entity_type] = []
                        if entity_text not in entities[entity_type]:
                            entities[entity_type].append(entity_text)
            
            return entities
            
        except Exception as e:
            logger.warning(f"NLTK entity extraction failed: {e}")
            return {}
    
    def classify_content_type(self, text: str) -> str:
        """
        Classify content type based on keywords and structure.
        Useful for FDA drug labels.
        """
        text_lower = text.lower()
        
        # Drug label indicators
        if any(keyword in text_lower for keyword in [
            'indication', 'dosage', 'administration', 'contraindication',
            'warning', 'adverse reaction', 'drug interaction', 'overdosage'
        ]):
            return 'drug_label'
        
        # Clinical trial indicators
        if any(keyword in text_lower for keyword in [
            'clinical trial', 'efficacy', 'safety', 'placebo', 'double-blind'
        ]):
            return 'clinical_data'
        
        # Regulatory indicators
        if any(keyword in text_lower for keyword in [
            'fda', 'approval', 'regulation', 'compliance', 'inspection'
        ]):
            return 'regulatory'
        
        # SOP indicators
        if any(keyword in text_lower for keyword in [
            'procedure', 'protocol', 'standard operating', 'sop', 'step'
        ]):
            return 'sop'
        
        return 'general'