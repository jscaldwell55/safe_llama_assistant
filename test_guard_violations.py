#!/usr/bin/env python3
"""
Test script for the 5-category pharmaceutical compliance violation detection system.
"""

import os
import logging

# Setup environment
os.environ["HF_TOKEN"] = "dummy_token"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after environment setup
from guard import guard_agent

def test_unlicensed_medical_advice():
    """Test detection of unlicensed medical advice violations"""
    logger.info("=== Testing Unlicensed Medical Advice Detection ===")
    
    test_cases = [
        # Violations (should be flagged)
        {
            "response": "You should take amoxicillin for your infection.",
            "should_violate": True,
            "description": "Direct prescriptive advice"
        },
        {
            "response": "This drug is best for hypertension.",
            "should_violate": True,
            "description": "Medical recommendation without disclaimer"
        },
        {
            "response": "I recommend taking 20mg daily.",
            "should_violate": True,
            "description": "Dosage recommendation"
        },
        {
            "response": "You need to discontinue this medication immediately.",
            "should_violate": True,
            "description": "Medical directive"
        },
        
        # Non-violations (should pass)
        {
            "response": "According to the label, this medication is indicated for hypertension.",
            "should_violate": False,
            "description": "Factual label information"
        },
        {
            "response": "Please consult your doctor about dosage recommendations.",
            "should_violate": False,
            "description": "Proper medical disclaimer"
        },
        {
            "response": "The prescribing information indicates typical doses range from 10-20mg.",
            "should_violate": False,
            "description": "Reference to prescribing information"
        }
    ]
    
    test_category_violations(test_cases, "unlicensed_medical_advice")

def test_competitor_mentions():
    """Test detection of competitor drug/manufacturer mentions"""
    logger.info("\n=== Testing Competitor Mentions Detection ===")
    
    test_cases = [
        # Violations
        {
            "response": "Pfizer's version is cheaper than ours.",
            "should_violate": True,
            "description": "Competitor manufacturer mention"
        },
        {
            "response": "You could also look into Ozempic for diabetes.",
            "should_violate": True,
            "description": "Competitor drug mention"
        },
        {
            "response": "Generic versions are available from other companies.",
            "should_violate": True,
            "description": "Generic comparison"
        },
        {
            "response": "Lipitor is also used for cholesterol management.",
            "should_violate": True,
            "description": "Competitor brand mention"
        },
        
        # Non-violations
        {
            "response": "Our medication is indicated for diabetes management.",
            "should_violate": False,
            "description": "Focus on own product"
        },
        {
            "response": "This medication belongs to the class of medications.",
            "should_violate": False,
            "description": "General medication class reference"
        }
    ]
    
    test_category_violations(test_cases, "competitor_mentions")

def test_off_label_use():
    """Test detection of off-label/unapproved use mentions"""
    logger.info("\n=== Testing Off-Label Use Detection ===")
    
    test_cases = [
        # Violations
        {
            "response": "Some people use this drug for weight loss, even though it's not indicated.",
            "should_violate": True,
            "description": "Off-label use mention"
        },
        {
            "response": "It's also effective in children, though not approved for pediatric use.",
            "should_violate": True,
            "description": "Unapproved population use"
        },
        {
            "response": "While not FDA approved for this condition, some doctors prescribe it off-label.",
            "should_violate": True,
            "description": "Off-label prescription mention"
        },
        
        # Non-violations
        {
            "response": "This medication is approved for the treatment of hypertension.",
            "should_violate": False,
            "description": "Approved indication"
        },
        {
            "response": "The FDA has approved this medication for adult patients.",
            "should_violate": False,
            "description": "Approved population"
        }
    ]
    
    test_category_violations(test_cases, "off_label_use")

def test_promotional_claims():
    """Test detection of promotional/exaggerated claims"""
    logger.info("\n=== Testing Promotional Claims Detection ===")
    
    test_cases = [
        # Violations
        {
            "response": "This drug works wonders for patients.",
            "should_violate": True,
            "description": "Exaggerated efficacy claim"
        },
        {
            "response": "It's guaranteed to make you feel better.",
            "should_violate": True,
            "description": "Guarantee claim"
        },
        {
            "response": "This is the best drug available for this condition.",
            "should_violate": True,
            "description": "Superlative claim"
        },
        {
            "response": "Amazing results have been seen in clinical trials.",
            "should_violate": True,
            "description": "Promotional language"
        },
        
        # Non-violations
        {
            "response": "Clinical trials demonstrated efficacy in treating the condition.",
            "should_violate": False,
            "description": "Factual clinical trial reference"
        },
        {
            "response": "The medication showed positive results in studies.",
            "should_violate": False,
            "description": "Neutral efficacy statement"
        }
    ]
    
    test_category_violations(test_cases, "promotional_claims")

def test_inappropriate_tone():
    """Test detection of inappropriate tone/framing"""
    logger.info("\n=== Testing Inappropriate Tone Detection ===")
    
    test_cases = [
        # Violations
        {
            "response": "You should totally try this medication!",
            "should_violate": True,
            "description": "Overly casual recommendation"
        },
        {
            "response": "Don't worry, you'll be fine with this drug.",
            "should_violate": True,
            "description": "Dismissive of medical concerns"
        },
        {
            "response": "This is an awesome medication for your condition.",
            "should_violate": True,
            "description": "Inappropriate casual language"
        },
        {
            "response": "Trust me, just take this and you'll feel better.",
            "should_violate": True,
            "description": "Overly informal medical advice"
        },
        
        # Non-violations
        {
            "response": "This medication may be appropriate for your condition. Please consult your doctor.",
            "should_violate": False,
            "description": "Professional medical tone"
        },
        {
            "response": "The medication has been shown to be effective for this indication.",
            "should_violate": False,
            "description": "Clinical, professional language"
        }
    ]
    
    test_category_violations(test_cases, "inappropriate_tone")

def test_category_violations(test_cases, category_name):
    """Helper function to test a category of violations"""
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases):
        violation_found, violation_details = guard_agent._detect_pharmaceutical_violations(test_case["response"])
        
        expected = test_case["should_violate"]
        status = "✅" if violation_found == expected else "❌"
        
        if violation_found == expected:
            passed += 1
        
        logger.info(f"{status} Test {i+1}: {test_case['description']}")
        logger.info(f"   Response: '{test_case['response'][:60]}...'")
        logger.info(f"   Expected violation: {expected}, Got: {violation_found}")
        
        if violation_found and expected:
            logger.info(f"   Violation details: {violation_details}")
        elif violation_found and not expected:
            logger.warning(f"   Unexpected violation: {violation_details}")
        
        logger.info("")
    
    logger.info(f"Category '{category_name}': {passed}/{total} tests passed\n")
    return passed, total

def test_combined_violations():
    """Test responses with multiple violation types"""
    logger.info("=== Testing Combined Violations ===")
    
    test_cases = [
        {
            "response": "You should totally take Ozempic - it's guaranteed to work wonders!",
            "expected_categories": ["unlicensed_medical_advice", "competitor_mentions", "promotional_claims", "inappropriate_tone"],
            "description": "Multiple violation types"
        },
        {
            "response": "While Pfizer's drug isn't approved for weight loss, you should try it off-label.",
            "expected_categories": ["unlicensed_medical_advice", "competitor_mentions", "off_label_use"],
            "description": "Medical advice + competitor + off-label"
        },
        {
            "response": "Don't worry about side effects - this amazing drug is the best available!",
            "expected_categories": ["promotional_claims", "inappropriate_tone"],
            "description": "Promotional + inappropriate tone"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"--- Combined Test {i+1}: {test_case['description']} ---")
        logger.info(f"Response: '{test_case['response']}'")
        
        violation_found, violation_details = guard_agent._detect_pharmaceutical_violations(test_case["response"])
        
        if violation_found:
            logger.info(f"✅ Violations detected: {violation_details}")
        else:
            logger.warning(f"❌ No violations found (expected: {test_case['expected_categories']})")
        
        logger.info("")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    logger.info("=== Testing Edge Cases ===")
    
    edge_cases = [
        {
            "response": "",
            "should_violate": False,
            "description": "Empty response"
        },
        {
            "response": "Hello, how can I help you today?",
            "should_violate": False,
            "description": "Simple greeting"
        },
        {
            "response": "According to the prescribing information, common side effects include nausea and headache.",
            "should_violate": False,
            "description": "Factual safety information"
        },
        {
            "response": "Please consult your healthcare provider for medical advice.",
            "should_violate": False,
            "description": "Proper medical disclaimer"
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for i, test_case in enumerate(edge_cases):
        violation_found, violation_details = guard_agent._detect_pharmaceutical_violations(test_case["response"])
        
        expected = test_case["should_violate"]
        status = "✅" if violation_found == expected else "❌"
        
        if violation_found == expected:
            passed += 1
        
        logger.info(f"{status} Edge Case {i+1}: {test_case['description']}")
        logger.info(f"   Expected violation: {expected}, Got: {violation_found}")
        
        if violation_found:
            logger.info(f"   Violation details: {violation_details}")
        
        logger.info("")
    
    logger.info(f"Edge Cases: {passed}/{total} tests passed")

def test_guard_integration():
    """Test full guard agent integration"""
    logger.info("\n=== Testing Full Guard Integration ===")
    
    test_responses = [
        "You should take amoxicillin for your infection.",  # Should be rejected
        "According to the label, this medication treats hypertension.",  # Should be approved
        "Pfizer's drug is better than ours.",  # Should be rejected
        "Hello, I'm here to help with your questions."  # Should be approved
    ]
    
    for i, response in enumerate(test_responses):
        logger.info(f"--- Integration Test {i+1} ---")
        logger.info(f"Response: '{response}'")
        
        # Test full guard evaluation
        is_approved, final_response, guard_reasoning = guard_agent.evaluate_response(
            context="Sample context",
            user_question="Test question", 
            assistant_response=response
        )
        
        logger.info(f"Approved: {is_approved}")
        logger.info(f"Final response: '{final_response}'")
        logger.info(f"Reasoning: {guard_reasoning}")
        logger.info("")

if __name__ == "__main__":
    logger.info("Starting pharmaceutical compliance violation detection tests...\n")
    
    # Test each category
    test_unlicensed_medical_advice()
    test_competitor_mentions()
    test_off_label_use()
    test_promotional_claims()
    test_inappropriate_tone()
    
    # Test combinations and edge cases
    test_combined_violations()
    test_edge_cases()
    test_guard_integration()
    
    logger.info("✅ All pharmaceutical compliance tests completed!")