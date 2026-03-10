"""
Test script for the hybrid claim detector (regex + NLI).

Tests the ClaimDetector with various sentence types:
- Factual claims (should be True)
- Jokes/humor (should be False with NLI)
- Opinions (should be False with NLI)
- Fillers/meta (should be False via regex fast-pass)
- Edge cases
"""

import sys
import io

# Fix Windows console encoding for unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r'd:\Github Repositories\semantic-entropy-probes\backend')

from claim_filter import ClaimDetector

def run_tests():
    print("=" * 80)
    print("CLAIM DETECTOR TEST SUITE")
    print("=" * 80)
    
    # Create detector and load NLI model
    detector = ClaimDetector()
    nli_loaded = detector.load_nli_model()
    
    print(f"\nNLI Model loaded: {nli_loaded}")
    print(f"NLI Available: {detector.nli_available}")
    print()
    
    # Test cases: (sentence, expected_is_claim, category)
    test_cases = [
        # === Factual claims (should be True) ===
        ("The capital of France is Paris.", True, "FACT"),
        ("Water boils at 100 degrees Celsius.", True, "FACT"),
        ("The Earth orbits the Sun every 365.25 days.", True, "FACT"),
        ("Python was created by Guido van Rossum in 1991.", True, "FACT"),
        
        # === Jokes / humor (should be False with NLI) ===
        ("You're so old, I'm starting to think your phone's autocorrect is just a nice way of saying I have no idea what you said.", False, "JOKE"),
        ("Why did the chicken cross the road? To get to the other side.", False, "JOKE"),
        ("You're so slow, even your Wi-Fi gives up on you.", False, "JOKE"),
        ("I told my computer I needed a break, and it froze.", False, "JOKE"),
        
        # === Opinions (should be False with NLI) ===
        ("Pizza is the best food ever invented.", False, "OPINION"),
        ("That movie was absolutely terrible.", False, "OPINION"),
        
        # === Fillers/meta (should be False via regex) ===
        ("Sure!", False, "FILLER"),
        ("Let me explain this to you.", False, "META"),
        ("Here's what you need to know:", False, "META"),
        ("Hope this helps!", False, "META"),
        ("Great question!", False, "FILLER"),
        
        # === Creative/subjective (should be False with NLI) ===
        ("Life is like a box of chocolates.", False, "CREATIVE"),
        ("Roses are red, violets are blue.", False, "CREATIVE"),
    ]
    
    passed = 0
    failed = 0
    
    for sentence, expected, category in test_cases:
        result = detector.is_claim(sentence)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            failed += 1
        else:
            passed += 1
        
        short = sentence[:65] + ("..." if len(sentence) > 65 else "")
        print(f"[{status}] [{category:8s}] is_claim={str(result):5s} (exp={str(expected):5s}) | \"{short}\"")
    
    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    run_tests()
