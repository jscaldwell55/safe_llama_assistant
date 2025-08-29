# conversation_flow.py - Intelligent Conversation Flow Management

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ClarificationType(Enum):
    DOSAGE = "dosage"
    SIDE_EFFECTS = "side_effects"
    INTERACTIONS = "interactions"
    PATIENT_GROUP = "patient_group"
    ADMINISTRATION = "administration"
    TIMING = "timing"
    SEVERITY = "severity"

@dataclass
class ClarificationRequest:
    """Represents a clarification needed from the user"""
    clarification_type: ClarificationType
    question: str
    options: List[str]
    context: str
    confidence: float

class ConversationFlowManager:
    """
    Manages intelligent conversation flows and clarifications
    """
    
    def __init__(self):
        self.clarification_templates = {
            ClarificationType.DOSAGE: {
                "questions": [
                    "Are you asking about dosage for adults or children?",
                    "Are you interested in the standard dose or maximum dose?",
                    "Do you need information about oral or injectable forms?"
                ],
                "options": {
                    "age_group": ["Adults", "Children", "Elderly", "All age groups"],
                    "dose_type": ["Standard dose", "Maximum dose", "Starting dose", "Maintenance dose"],
                    "form": ["Oral tablets", "Injection", "Liquid suspension", "All forms"]
                }
            },
            ClarificationType.SIDE_EFFECTS: {
                "questions": [
                    "Are you interested in common or serious side effects?",
                    "Do you want to know about specific side effects or a general overview?",
                    "Are you concerned about side effects for a particular patient group?"
                ],
                "options": {
                    "severity": ["Common side effects", "Serious side effects", "All side effects"],
                    "specificity": ["General overview", "Specific symptoms", "Frequency data"],
                    "patient_group": ["General population", "Elderly", "Children", "Pregnancy"]
                }
            },
            ClarificationType.INTERACTIONS: {
                "questions": [
                    "Are you asking about interactions with other medications or substances?",
                    "Do you have a specific medication or class of drugs in mind?",
                    "Are you interested in food interactions or drug interactions?"
                ],
                "options": {
                    "interaction_type": ["Drug interactions", "Food interactions", "Alcohol", "All interactions"],
                    "specificity": ["Specific drug", "Drug class", "General overview"],
                    "severity": ["Major interactions", "Moderate interactions", "All interactions"]
                }
            },
            ClarificationType.PATIENT_GROUP: {
                "questions": [
                    "Which patient group are you asking about?",
                    "Do you need information for patients with specific conditions?",
                ],
                "options": {
                    "group": ["Adults", "Children", "Elderly", "Pregnant women", "Nursing mothers"],
                    "conditions": ["Kidney disease", "Liver disease", "Heart disease", "No specific conditions"]
                }
            },
            ClarificationType.ADMINISTRATION: {
                "questions": [
                    "How is the medication being administered?",
                    "Do you need instructions for a specific route of administration?"
                ],
                "options": {
                    "route": ["Oral", "Injection", "Topical", "Inhalation"],
                    "timing": ["With food", "Empty stomach", "Specific time of day", "No preference"]
                }
            }
        }
        
        # Track conversation state
        self.pending_clarifications: List[ClarificationRequest] = []
        self.clarification_history: List[Tuple[str, str]] = []  # (question, answer) pairs
    
    def analyze_ambiguity(self, query: str, entities: Dict) -> List[ClarificationRequest]:
        """
        Analyze query for ambiguities that need clarification
        """
        clarifications = []
        query_lower = query.lower()
        
        # Check for dosage ambiguity
        if self._needs_dosage_clarification(query_lower, entities):
            clarifications.append(self._create_dosage_clarification(query_lower))
        
        # Check for side effects ambiguity
        if self._needs_side_effects_clarification(query_lower, entities):
            clarifications.append(self._create_side_effects_clarification(query_lower))
        
        # Check for interactions ambiguity
        if self._needs_interactions_clarification(query_lower, entities):
            clarifications.append(self._create_interactions_clarification(query_lower))
        
        # Check for patient group ambiguity
        if self._needs_patient_group_clarification(query_lower, entities):
            clarifications.append(self._create_patient_group_clarification(query_lower))
        
        return clarifications
    
    def _needs_dosage_clarification(self, query: str, entities: Dict) -> bool:
        """Check if dosage query needs clarification"""
        dosage_keywords = ["dose", "dosage", "how much", "amount", "quantity"]
        has_dosage_query = any(keyword in query for keyword in dosage_keywords)
        
        if has_dosage_query:
            # Check if patient group is specified
            age_specified = any(term in query for term in ["adult", "child", "elderly", "pediatric"])
            form_specified = any(term in query for term in ["oral", "injection", "tablet", "liquid"])
            
            return not (age_specified and form_specified)
        
        return False
    
    def _needs_side_effects_clarification(self, query: str, entities: Dict) -> bool:
        """Check if side effects query needs clarification"""
        se_keywords = ["side effect", "adverse", "reaction", "symptom"]
        has_se_query = any(keyword in query for keyword in se_keywords)
        
        if has_se_query:
            # Check if severity is specified
            severity_specified = any(term in query for term in ["common", "serious", "severe", "mild"])
            return not severity_specified
        
        return False
    
    def _needs_interactions_clarification(self, query: str, entities: Dict) -> bool:
        """Check if interactions query needs clarification"""
        int_keywords = ["interact", "combine", "mix", "together with"]
        has_int_query = any(keyword in query for keyword in int_keywords)
        
        if has_int_query:
            # Check if specific drug or type is mentioned
            specific_drug = entities.get("has_drug_mention", False) and "journvax" not in query
            type_specified = any(term in query for term in ["food", "alcohol", "drug", "medication"])
            
            return not (specific_drug or type_specified)
        
        return False
    
    def _needs_patient_group_clarification(self, query: str, entities: Dict) -> bool:
        """Check if patient group needs clarification"""
        # Look for general safety/use questions without specific group
        safety_keywords = ["safe", "can take", "appropriate for", "suitable"]
        has_safety_query = any(keyword in query for keyword in safety_keywords)
        
        if has_safety_query:
            group_specified = any(term in query for term in 
                                ["adult", "child", "elderly", "pregnant", "nursing", "kidney", "liver"])
            return not group_specified
        
        return False
    
    def _create_dosage_clarification(self, query: str) -> ClarificationRequest:
        """Create dosage clarification request"""
        templates = self.clarification_templates[ClarificationType.DOSAGE]
        
        # Choose appropriate question based on query
        if "child" in query or "pediatric" in query:
            question = "What age group specifically?"
            options = ["Infants (0-1 year)", "Children (2-11 years)", "Adolescents (12-17 years)"]
        else:
            question = templates["questions"][0]
            options = templates["options"]["age_group"]
        
        return ClarificationRequest(
            clarification_type=ClarificationType.DOSAGE,
            question=question,
            options=options,
            context=query,
            confidence=0.8
        )
    
    def _create_side_effects_clarification(self, query: str) -> ClarificationRequest:
        """Create side effects clarification request"""
        templates = self.clarification_templates[ClarificationType.SIDE_EFFECTS]
        
        return ClarificationRequest(
            clarification_type=ClarificationType.SIDE_EFFECTS,
            question=templates["questions"][0],
            options=templates["options"]["severity"],
            context=query,
            confidence=0.7
        )
    
    def _create_interactions_clarification(self, query: str) -> ClarificationRequest:
        """Create interactions clarification request"""
        templates = self.clarification_templates[ClarificationType.INTERACTIONS]
        
        return ClarificationRequest(
            clarification_type=ClarificationType.INTERACTIONS,
            question=templates["questions"][0],
            options=templates["options"]["interaction_type"],
            context=query,
            confidence=0.7
        )
    
    def _create_patient_group_clarification(self, query: str) -> ClarificationRequest:
        """Create patient group clarification request"""
        templates = self.clarification_templates[ClarificationType.PATIENT_GROUP]
        
        return ClarificationRequest(
            clarification_type=ClarificationType.PATIENT_GROUP,
            question=templates["questions"][0],
            options=templates["options"]["group"],
            context=query,
            confidence=0.6
        )
    
    def format_clarification_response(self, clarifications: List[ClarificationRequest]) -> str:
        """Format clarification requests for user display"""
        
        if not clarifications:
            return ""
        
        # Take the highest confidence clarification
        clarification = max(clarifications, key=lambda c: c.confidence)
        
        response_parts = [
            "I'd be happy to help with that information. To provide the most relevant details:",
            "",
            clarification.question,
            ""
        ]
        
        # Add numbered options
        for i, option in enumerate(clarification.options, 1):
            response_parts.append(f"{i}. {option}")
        
        response_parts.append("")
        response_parts.append("Please let me know which option best matches what you're looking for, or feel free to rephrase your question.")
        
        return "\n".join(response_parts)
    
    def refine_query_with_clarification(self, original_query: str, clarification_answer: str) -> str:
        """
        Refine the original query based on clarification answer
        """
        # Store in history
        self.clarification_history.append((original_query, clarification_answer))
        
        # Build refined query
        refined_parts = [original_query]
        
        # Add clarification context
        clarification_lower = clarification_answer.lower()
        
        # Add specific context based on answer
        if "adult" in clarification_lower:
            refined_parts.append("for adults")
        elif "child" in clarification_lower:
            refined_parts.append("for children")
        elif "common" in clarification_lower:
            refined_parts.append("focusing on common occurrences")
        elif "serious" in clarification_lower:
            refined_parts.append("focusing on serious cases")
        elif "drug interaction" in clarification_lower:
            refined_parts.append("specifically about drug interactions")
        elif "food" in clarification_lower:
            refined_parts.append("specifically about food interactions")
        
        refined_query = " ".join(refined_parts)
        
        logger.info(f"Refined query: '{original_query}' -> '{refined_query}'")
        
        return refined_query
    
    def should_ask_clarification(self, clarifications: List[ClarificationRequest], 
                                retrieval_score: float) -> bool:
        """
        Decide whether to ask for clarification based on confidence and retrieval quality
        """
        if not clarifications:
            return False
        
        # If retrieval score is very high, probably don't need clarification
        if retrieval_score > 0.7:
            return False
        
        # If we have high-confidence clarification needs and medium retrieval
        max_confidence = max(c.confidence for c in clarifications)
        if max_confidence > 0.7 and retrieval_score < 0.5:
            return True
        
        # If multiple clarifications needed
        if len(clarifications) >= 2:
            return True
        
        return False

# Singleton instance
_flow_manager_instance: Optional[ConversationFlowManager] = None

def get_flow_manager() -> ConversationFlowManager:
    """Get singleton conversation flow manager"""
    global _flow_manager_instance
    if _flow_manager_instance is None:
        _flow_manager_instance = ConversationFlowManager()
    return _flow_manager_instance