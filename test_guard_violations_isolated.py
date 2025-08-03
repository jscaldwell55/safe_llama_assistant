#!/usr/bin/env python3
"""
Isolated test for pharmaceutical compliance violation detection without external dependencies.
"""

import os
import logging
import re
from typing import Dict, List, Tuple

# Setup environment
os.environ["HF_TOKEN"] = "dummy_token"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedGuardAgent:
    """Simplified guard agent for testing violation detection"""
    
    def __init__(self):
        self.fallback_message = "I can't help with that. Can we talk about something else?"
        
        # 5-Category Violation Detection System
        self.violation_categories = {
            "unlicensed_medical_advice": {
                "patterns": [
                    r'\byou should take\b',
                    r'\bi recommend (taking|using)\b',
                    r'\bthis drug is best for\b',
                    r'\btake \d+mg\b',
                    r'\bstart with\b.*\bmg\b',
                    r'\bdiscontinue\b.*\bmedication\b',
                    r'\byou need to\b.*\b(drug|medication|prescription)\b',
                    r'\bdiagnosed with\b',
                    r'\byou have\b.*\b(condition|disease|disorder)\b'
                ]
            },
            "competitor_mentions": {
                "patterns": [
                    r'\b(pfizer|merck|bristol myers|novartis|roche|astrazeneca|gsk|sanofi|eli lilly|abbvie|amgen|gilead)\b',
                    r'\b(ozempic|wegovy|mounjaro|trulicity|januvia|victoza|byetta)\b',
                    r'\b(lipitor|crestor|zocor|pravachol|livalo)\b',
                    r'\bgeneric version\b',
                    r'\bcompetitor\b.*\b(drug|product)\b',
                    r'\bother brands?\b',
                    r'\balternative products?\b'
                ]
            },
            "off_label_use": {
                "patterns": [
                    r'\boff.label\b',
                    r'\bnot approved for\b',
                    r'\bsome people use.*for\b',
                    r'\balso effective (in|for)\b.*\bnot approved\b',
                    r'\bunlabeled use\b',
                    r'\bnot indicated for\b.*\bbut\b',
                    r'\bpediatric use\b.*\bnot approved\b',
                    r'\bweight loss\b.*\bnot indicated\b'
                ]
            },
            "promotional_claims": {
                "patterns": [
                    r'\bworks wonders\b',
                    r'\bguaranteed to\b',
                    r'\bbest drug\b',
                    r'\bmost effective\b',
                    r'\bsuperior to\b',
                    r'\bbetter than\b.*\b(other|all)\b',
                    r'\bperfect for\b',
                    r'\bamazing results?\b',
                    r'\bincredible\b.*\b(results?|effects?)\b',
                    r'\blife.changing\b'
                ]
            },
            "inappropriate_tone": {
                "patterns": [
                    r'\byou should totally\b',
                    r'\bdon\'t worry\b',
                    r'\byou\'ll be fine\b',
                    r'\bno big deal\b',
                    r'\bawesome\b.*\b(drug|medication)\b',
                    r'\bsuper effective\b',
                    r'\bgo for it\b',
                    r'\bwhy not try\b',
                    r'\bjust take\b',
                    r'\btrust me\b'
                ]
            }
        }
    
    def detect_pharmaceutical_violations(self, assistant_response: str) -> Tuple[bool, str]:
        """Detect pharmaceutical compliance violations in the response"""
        response_lower = assistant_response.lower()
        violations_found = []
        
        for category, detection_config in self.violation_categories.items():
            patterns = detection_config["patterns"]
            
            # Check regex patterns
            for pattern in patterns:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    violations_found.append({
                        "category": category,
                        "pattern": pattern,
                        "type": "pattern_match"
                    })
            
            # Additional context-aware checks
            category_violations = self.check_category_specific_violations(category, response_lower)
            violations_found.extend(category_violations)
        
        if violations_found:
            violation_details = self.format_violation_details(violations_found)
            return True, violation_details
        
        return False, "No violations detected"
    
    def check_category_specific_violations(self, category: str, response_lower: str) -> List[Dict]:
        """Check for category-specific violations with additional context"""
        violations = []
        
        if category == "unlicensed_medical_advice":
            # Check for directive language about medications
            if any(phrase in response_lower for phrase in [
                "should take", "need to take", "must take", "start taking",
                "stop taking", "increase dose", "decrease dose"
            ]):
                violations.append({
                    "category": category,
                    "type": "directive_medical_language",
                    "description": "Contains directive medical advice"
                })
        
        elif category == "competitor_mentions":
            # Check for brand name mentions (expanded list)
            competitor_brands = [
                "advil", "tylenol", "aleve", "motrin", "excedrin",
                "prozac", "zoloft", "wellbutrin", "cymbalta", "effexor",
                "nexium", "prilosec", "prevacid", "pepcid", "zantac"
            ]
            for brand in competitor_brands:
                if brand in response_lower:
                    violations.append({
                        "category": category,
                        "type": "competitor_brand_mention",
                        "brand": brand,
                        "description": f"Mentions competitor brand: {brand}"
                    })
        
        elif category == "off_label_use":
            # Check for off-label indicators
            off_label_indicators = [
                "not fda approved for", "not approved for this use",
                "off-label", "unlabeled use", "investigational use"
            ]
            for indicator in off_label_indicators:
                if indicator in response_lower:
                    violations.append({
                        "category": category,
                        "type": "off_label_reference",
                        "description": f"References off-label use: {indicator}"
                    })
        
        elif category == "promotional_claims":
            # Check for superlative and guarantee language
            promotional_phrases = [
                "best", "greatest", "most effective", "superior",
                "guaranteed", "proven to work", "amazing results",
                "miracle", "breakthrough", "revolutionary"
            ]
            for phrase in promotional_phrases:
                if phrase in response_lower:
                    violations.append({
                        "category": category,
                        "type": "promotional_language",
                        "phrase": phrase,
                        "description": f"Uses promotional language: {phrase}"
                    })
        
        elif category == "inappropriate_tone":
            # Check for casual/informal medical advice
            casual_phrases = [
                "totally", "super", "awesome", "no worries",
                "you'll be fine", "don't stress", "go for it",
                "trust me", "just try it"
            ]
            for phrase in casual_phrases:
                if phrase in response_lower:
                    violations.append({
                        "category": category,
                        "type": "inappropriate_casualness",
                        "phrase": phrase,
                        "description": f"Inappropriate casual tone: {phrase}"
                    })
        
        return violations
    
    def format_violation_details(self, violations: List[Dict]) -> str:
        """Format violation details for logging and debugging"""
        if not violations:
            return "No violations found"
        
        # Group by category
        by_category = {}
        for violation in violations:
            category = violation["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(violation)
        
        # Format summary
        summary_parts = []
        for category, category_violations in by_category.items():
            category_name = category.replace("_", " ").title()
            summary_parts.append(f"{category_name} ({len(category_violations)} violations)")
        
        return f"Pharmaceutical compliance violations detected: {', '.join(summary_parts)}"

def test_unlicensed_medical_advice():
    """Test detection of unlicensed medical advice violations"""
    logger.info("=== Testing Unlicensed Medical Advice Detection ===")
    
    guard = SimplifiedGuardAgent()
    
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
        {
            "response": "Take 10mg twice daily for best results.",
            "should_violate": True,
            "description": "Specific dosing instruction"
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
    
    return test_category_violations(guard, test_cases, "unlicensed_medical_advice")

def test_competitor_mentions():
    """Test detection of competitor drug/manufacturer mentions"""
    logger.info("\n=== Testing Competitor Mentions Detection ===")
    
    guard = SimplifiedGuardAgent()
    
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
        {
            "response": "Many patients also use Advil for pain relief.",
            "should_violate": True,
            "description": "OTC competitor mention"
        },
        
        # Non-violations
        {
            "response": "Our medication is indicated for diabetes management.",
            "should_violate": False,
            "description": "Focus on own product"
        },
        {
            "response": "This medication belongs to the ACE inhibitor class.",
            "should_violate": False,
            "description": "General medication class reference"
        }
    ]
    
    return test_category_violations(guard, test_cases, "competitor_mentions")

def test_off_label_use():
    """Test detection of off-label/unapproved use mentions"""
    logger.info("\n=== Testing Off-Label Use Detection ===")
    
    guard = SimplifiedGuardAgent()
    
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
        {
            "response": "This medication has unlabeled uses for anxiety disorders.",
            "should_violate": True,
            "description": "Unlabeled use mention"
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
    
    return test_category_violations(guard, test_cases, "off_label_use")

def test_promotional_claims():
    """Test detection of promotional/exaggerated claims"""
    logger.info("\n=== Testing Promotional Claims Detection ===")
    
    guard = SimplifiedGuardAgent()
    
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
        {
            "response": "This revolutionary breakthrough will change your life.",
            "should_violate": True,
            "description": "Multiple promotional terms"
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
    
    return test_category_violations(guard, test_cases, "promotional_claims")

def test_inappropriate_tone():
    """Test detection of inappropriate tone/framing"""
    logger.info("\n=== Testing Inappropriate Tone Detection ===")
    
    guard = SimplifiedGuardAgent()
    
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
        {
            "response": "Go for it - this medication is super effective!",
            "should_violate": True,
            "description": "Casual encouragement with promotional language"
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
    
    return test_category_violations(guard, test_cases, "inappropriate_tone")

def test_category_violations(guard, test_cases, category_name):
    """Helper function to test a category of violations"""
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases):
        violation_found, violation_details = guard.detect_pharmaceutical_violations(test_case["response"])
        
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
    
    guard = SimplifiedGuardAgent()
    
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
        
        violation_found, violation_details = guard.detect_pharmaceutical_violations(test_case["response"])
        
        if violation_found:
            logger.info(f"✅ Violations detected: {violation_details}")
        else:
            logger.warning(f"❌ No violations found (expected: {test_case['expected_categories']})")
        
        logger.info("")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    logger.info("=== Testing Edge Cases ===")
    
    guard = SimplifiedGuardAgent()
    
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
        },
        {
            "response": "This medication is indicated for the treatment of hypertension in adults.",
            "should_violate": False,
            "description": "Factual indication statement"
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for i, test_case in enumerate(edge_cases):
        violation_found, violation_details = guard.detect_pharmaceutical_violations(test_case["response"])
        
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
    return passed, total

if __name__ == "__main__":
    logger.info("Starting pharmaceutical compliance violation detection tests...\n")
    
    # Track overall results
    total_passed = 0
    total_tests = 0
    
    # Test each category
    passed, tests = test_unlicensed_medical_advice()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_competitor_mentions()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_off_label_use()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_promotional_claims()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_inappropriate_tone()
    total_passed += passed
    total_tests += tests
    
    # Test combinations and edge cases
    test_combined_violations()
    
    passed, tests = test_edge_cases()
    total_passed += passed
    total_tests += tests
    
    # Final summary
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL RESULTS: {total_passed}/{total_tests} tests passed ({success_rate:.1f}%)")
    logger.info(f"{'='*50}")
    
    if success_rate >= 90:
        logger.info("✅ Pharmaceutical compliance system is working excellently!")
    elif success_rate >= 80:
        logger.info("✅ Pharmaceutical compliance system is working well!")
    else:
        logger.warning("⚠️ Pharmaceutical compliance system needs improvement.")
    
    logger.info("\n✅ All pharmaceutical compliance tests completed!")