import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from rag import retrieve, rag_system
from conversation import conversation_manager
import re

logger = logging.getLogger(__name__)

class ConversationMode(Enum):
    """Different modes of conversation engagement"""
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    INFORMATION_REQUEST = "information_request"
    CHITCHAT = "chitchat"
    HELP = "help"
    SESSION_END = "session_end"

@dataclass
class ConversationResponse:
    """Structure for conversation responses"""
    text: str
    mode: ConversationMode
    has_rag_content: bool
    confidence: float
    follow_up_suggestions: List[str] = None
    debug_info: Dict[str, Any] = None

class ConversationalAgent:
    """
    Enhanced conversational agent that engages organically while respecting RAG-only constraints.
    
    Key principles:
    1. Engage naturally in conversation flow
    2. Only provide factual information from RAG content
    3. Use fallback responses when no RAG content available
    4. Maintain conversational context and continuity
    """
    
    def __init__(self):
        self.fallback_message = "I'm sorry, I don't seem to have any information on that. Can I help you with something else?"
        
        # Conversational elements that don't require RAG content
        self.social_patterns = {
            'greetings': [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\bhow are you\b',
                r'\bnice to meet you\b'
            ],
            'thanks': [
                r'\b(thank you|thanks|appreciate)\b',
                r'\bthat helps?\b',
                r'\bthat\'s helpful\b'
            ],
            'politeness': [
                r'\bplease\b',
                r'\bif you (don\'t mind|could)\b',
                r'\bwould you mind\b'
            ],
            'clarification_requests': [
                r'\bcan you (explain|clarify|elaborate)\b',
                r'\bwhat do you mean\b',
                r'\bi don\'t understand\b',
                r'\bcould you repeat\b'
            ],
            'affirmations': [
                r'\b(yes|yeah|ok|okay|sure|right|correct)\b',
                r'\bthat makes sense\b',
                r'\bi see\b'
            ]
        }
        
        # Follow-up question patterns
        self.follow_up_patterns = [
            r'\bwhat about\b',
            r'\band what\b',
            r'\balso\b',
            r'\btoo\b',
            r'\bas well\b',
            r'\bmore about\b',
            r'\bother\b',
            r'\belse\b',
            r'\bfurther\b'
        ]
        
        # Question starters that often need RAG content
        self.information_patterns = [
            r'\bwhat is\b',
            r'\bwho is\b',
            r'\bhow does\b',
            r'\bwhen did\b',
            r'\bwhere can\b',
            r'\bwhy does\b',
            r'\btell me about\b',
            r'\bexplain\b',
            r'\bdescribe\b',
            r'\bdefine\b'
        ]
    
    def classify_conversation_mode(self, query: str, has_rag_content: bool = None) -> ConversationMode:
        """Classify the type of conversational engagement needed"""
        query_lower = query.lower().strip()
        
        # Check for greetings
        if self._matches_patterns(query_lower, self.social_patterns['greetings']):
            return ConversationMode.GREETING
            
        # Check for help requests
        if any(word in query_lower for word in ['help', 'what can you do', 'capabilities', 'how to use']):
            return ConversationMode.HELP
            
        # Check for clarification requests
        if self._matches_patterns(query_lower, self.social_patterns['clarification_requests']):
            return ConversationMode.CLARIFICATION
            
        # Check for follow-up questions
        if self._matches_patterns(query_lower, self.follow_up_patterns) or conversation_manager.is_follow_up_question(query):
            return ConversationMode.FOLLOW_UP
            
        # Check for information requests
        if self._matches_patterns(query_lower, self.information_patterns) or '?' in query:
            return ConversationMode.INFORMATION_REQUEST
            
        # Default to chitchat for everything else
        return ConversationMode.CHITCHAT
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given regex patterns"""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def generate_greeting_response(self) -> ConversationResponse:
        """Generate a natural greeting response"""
        greetings = [
            "Hello! I'm here to help you find information from our knowledge base. What would you like to know?",
            "Hi there! I can assist you with questions about our documentation and data. What can I help you with?",
            "Good to see you! I'm ready to help you explore our knowledge base. What's on your mind?",
            "Hello! I'm your assistant for finding information in our enterprise knowledge base. How can I help today?"
        ]
        
        import random
        greeting_text = random.choice(greetings)
        
        return ConversationResponse(
            text=greeting_text,
            mode=ConversationMode.GREETING,
            has_rag_content=False,
            confidence=1.0,
            follow_up_suggestions=["Ask me about any topic in our knowledge base", "I can search our documentation for you"]
        )
    
    def generate_help_response(self) -> ConversationResponse:
        """Generate help information"""
        help_text = """I'm here to help you find information from our knowledge base! Here's what I can do:

• **Answer questions** about our documentation and data
• **Search and retrieve** relevant information for you  
• **Provide insights** based on our enterprise content
• **Have conversations** about topics in our knowledge base

**Important:** I can only share information that's actually in our knowledge base. If I don't have information on a topic, I'll let you know honestly.

What would you like to explore?"""
        
        return ConversationResponse(
            text=help_text,
            mode=ConversationMode.HELP,
            has_rag_content=False,
            confidence=1.0,
            follow_up_suggestions=["Try asking about any topic you're interested in", "I can search our knowledge base for specific information"]
        )
    
    def generate_session_end_response(self) -> ConversationResponse:
        """Generate session ending response"""
        end_text = "This session is ending, please feel free to start a new conversation."
        
        return ConversationResponse(
            text=end_text,
            mode=ConversationMode.SESSION_END,
            has_rag_content=False,
            confidence=1.0,
            follow_up_suggestions=["Click 'New Conversation' to start fresh"]
        )
    
    def generate_clarification_response(self, original_query: str, context: str = None) -> ConversationResponse:
        """Generate clarification when the query is unclear"""
        if context:
            clarification_text = f"I want to make sure I understand correctly. Are you asking about {context}? Could you provide a bit more detail about what specifically you'd like to know?"
        else:
            clarification_text = "I want to help you find the right information. Could you clarify what specifically you're looking for? The more details you provide, the better I can assist you."
        
        return ConversationResponse(
            text=clarification_text,
            mode=ConversationMode.CLARIFICATION,
            has_rag_content=False,
            confidence=0.7,
            follow_up_suggestions=["Try rephrasing your question", "Add more specific details about what you're looking for"]
        )
    
    def generate_chitchat_response(self, query: str) -> ConversationResponse:
        """Handle general chitchat while staying within boundaries"""
        
        # Check for thanks/appreciation
        if self._matches_patterns(query.lower(), self.social_patterns['thanks']):
            responses = [
                "You're welcome! Is there anything else you'd like to know?",
                "Happy to help! What else can I assist you with?",
                "Glad I could help! Do you have any other questions?",
                "You're very welcome! Feel free to ask me anything else."
            ]
        
        # Check for affirmations
        elif self._matches_patterns(query.lower(), self.social_patterns['affirmations']):
            responses = [
                "Great! Is there anything else you'd like to explore?",
                "Perfect! What else would you like to know?",
                "Excellent! Do you have any follow-up questions?",
                "Wonderful! How else can I help you?"
            ]
        
        # General chitchat
        else:
            responses = [
                "I'm here to help you find information from our knowledge base. What would you like to know?",
                "Let me help you explore our knowledge base. What topic interests you?",
                "I'd be happy to assist you with information from our documentation. What can I help you find?",
                "I'm ready to help you discover what's in our knowledge base. What would you like to learn about?"
            ]
        
        import random
        response_text = random.choice(responses)
        
        return ConversationResponse(
            text=response_text,
            mode=ConversationMode.CHITCHAT,
            has_rag_content=False,
            confidence=0.8,
            follow_up_suggestions=["Ask me about any topic in our knowledge base"]
        )
    
    def generate_follow_up_response(self, query: str, rag_content: List[str] = None) -> ConversationResponse:
        """Handle follow-up questions with context from previous conversation"""
        
        if not rag_content or len(rag_content) == 0:
            # No relevant content found for follow-up
            context = ""
            if conversation_manager.conversation and conversation_manager.conversation.current_topic:
                context = f" about {conversation_manager.conversation.current_topic}"
            
            response_text = f"I don't have additional information{context} in our knowledge base. Could you try asking about a different aspect, or would you like me to help you with something else?"
            
            return ConversationResponse(
                text=response_text,
                mode=ConversationMode.FOLLOW_UP,
                has_rag_content=False,
                confidence=0.6,
                follow_up_suggestions=["Try asking about a different topic", "Let me know if you need help with something else"]
            )
        
        # We have RAG content for the follow-up
        return ConversationResponse(
            text="",  # Will be filled by the main conversation handler
            mode=ConversationMode.FOLLOW_UP,
            has_rag_content=True,
            confidence=0.9
        )
    
    def generate_information_response(self, query: str, rag_content: List[str] = None) -> ConversationResponse:
        """Handle information requests"""
        
        if not rag_content or len(rag_content) == 0:
            return ConversationResponse(
                text=self.fallback_message,
                mode=ConversationMode.INFORMATION_REQUEST,
                has_rag_content=False,
                confidence=0.0,
                follow_up_suggestions=["Try asking about a different topic", "Rephrase your question with different keywords"]
            )
        
        # We have RAG content
        return ConversationResponse(
            text="",  # Will be filled by the main conversation handler
            mode=ConversationMode.INFORMATION_REQUEST,
            has_rag_content=True,
            confidence=0.9
        )
    
    def enhance_response_with_conversational_elements(self, base_response: str, mode: ConversationMode, query: str) -> str:
        """Add natural conversational elements to RAG-based responses"""
        
        # Add conversational starters based on mode
        if mode == ConversationMode.FOLLOW_UP:
            starters = [
                "Building on what we discussed, ",
                "To add to that, ",
                "Additionally, ",
                "Here's more information: "
            ]
        elif mode == ConversationMode.INFORMATION_REQUEST:
            starters = [
                "Based on our knowledge base, ",
                "Here's what I found: ",
                "According to our documentation, ",
                "From our available information, "
            ]
        else:
            starters = [""]
        
        import random
        starter = random.choice(starters) if starters else ""
        
        # Add conversational endings
        endings = [
            " Would you like to know more about any specific aspect?",
            " Is there anything else you'd like me to clarify?",
            " Do you have any follow-up questions?",
            " What else would you like to explore on this topic?"
        ]
        
        # Only add endings to longer responses
        ending = random.choice(endings) if len(base_response) > 50 else ""
        
        return f"{starter}{base_response}{ending}"
    
    def process_conversation(self, query: str) -> ConversationResponse:
        """Main conversation processing pipeline"""
        
        # Step 0: Check if session limit is reached BEFORE processing
        if conversation_manager.should_end_session():
            return self.generate_session_end_response()
        
        # Step 1: Classify conversation mode
        mode = self.classify_conversation_mode(query)
        
        # Step 2: Handle modes that don't require RAG content
        if mode == ConversationMode.GREETING:
            return self.generate_greeting_response()
        elif mode == ConversationMode.HELP:
            return self.generate_help_response()
        elif mode == ConversationMode.CHITCHAT:
            return self.generate_chitchat_response(query)
        
        # Step 3: For information/follow-up modes, try to get RAG content
        enhanced_query = conversation_manager.get_enhanced_query(query)
        rag_content = retrieve(enhanced_query)
        
        # Step 4: Handle clarification needs
        if mode == ConversationMode.CLARIFICATION:
            topic = conversation_manager.conversation.current_topic if conversation_manager.conversation else None
            return self.generate_clarification_response(query, topic)
        
        # Step 5: Handle follow-up questions
        if mode == ConversationMode.FOLLOW_UP:
            return self.generate_follow_up_response(query, rag_content)
        
        # Step 6: Handle information requests
        if mode == ConversationMode.INFORMATION_REQUEST:
            return self.generate_information_response(query, rag_content)
        
        # Default fallback
        return ConversationResponse(
            text=self.fallback_message,
            mode=mode,
            has_rag_content=False,
            confidence=0.0
        )

# Global conversational agent instance
conversational_agent = ConversationalAgent()