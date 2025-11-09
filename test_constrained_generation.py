#!/usr/bin/env python3
"""
Test constrained generation to verify it only produces valid labels
"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"
MODEL = "SmolLM2 1.7B"

# Test cases: text, labels, expected_valid_output
TEST_CASES = [
    {
        "name": "Single-token labels (sentiment)",
        "text": "This is amazing!",
        "labels": ["positive", "negative", "neutral"],
        "note": "Each label is a single token"
    },
    {
        "name": "Multi-token labels (cargo)",
        "text": "This container needs refrigeration",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"],
        "note": "Labels have 2-4 tokens each"
    },
    {
        "name": "Labels with shared prefixes",
        "text": "This is very urgent",
        "labels": ["urgent", "urgent escalation", "normal"],
        "note": "Tests prefix disambiguation"
    },
    {
        "name": "Mixed single and multi-token",
        "text": "I have a question about shipping",
        "labels": ["question", "complaint", "praise", "request", "information"],
        "note": "Intent classification with varied token counts"
    }
]

def test_classification(text: str, labels: list) -> dict:
    """Test classification and return result"""
    
    payload = {
        "text": text,
        "candidate_labels": labels,
        "model": MODEL,
        "use_logprobs": True,  # Enable logprobs for accurate confidence
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        result = None
        processing_time = None
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    if 'done' in data and data['done']:
                        result = data.get('result')
                        processing_time = data.get('processing_time')
                        break
                    if 'error' in data:
                        return {'success': False, 'error': data['error']}
                except json.JSONDecodeError:
                    continue
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'result': result,
            'processing_time': processing_time,
            'elapsed_time': elapsed
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


print("="*80)
print("CONSTRAINED GENERATION VALIDATION TEST")
print("="*80)
print(f"Model: {MODEL}")
print(f"Purpose: Verify constrained generation only produces valid candidate labels\n")

all_results = []
total_tests = len(TEST_CASES)
passed = 0
failed = 0

for i, test_case in enumerate(TEST_CASES, 1):
    print(f"\nTest {i}/{total_tests}: {test_case['name']}")
    print("-" * 80)
    print(f"Text: '{test_case['text']}'")
    print(f"Candidate Labels: {test_case['labels']}")
    print(f"Note: {test_case['note']}")
    
    result = test_classification(test_case['text'], test_case['labels'])
    
    if result['success']:
        res = result['result']
        top_label = res['top_label']
        
        # CRITICAL CHECK: Is top_label in candidate_labels?
        is_valid = top_label in test_case['labels']
        
        if is_valid:
            print(f"✓ PASS: Generated valid label '{top_label}'")
            print(f"  Confidence: {res['top_score']*100:.1f}%")
            print(f"  Processing Time: {result['processing_time']:.2f}s")
            passed += 1
        else:
            print(f"✗ FAIL: Generated invalid label '{top_label}'")
            print(f"  Expected one of: {test_case['labels']}")
            print(f"  This should NEVER happen with constrained generation!")
            failed += 1
        
        all_results.append({
            'test': test_case['name'],
            'valid': is_valid,
            'top_label': top_label,
            'all_labels': res['labels']
        })
    else:
        print(f"✗ ERROR: {result.get('error')}")
        failed += 1

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Total Tests: {total_tests}")
print(f"✓ Passed: {passed}/{total_tests} ({passed/total_tests*100:.1f}%)")
print(f"✗ Failed: {failed}/{total_tests}")

if failed == 0:
    print("\n🎉 SUCCESS: All tests passed!")
    print("Constrained generation correctly ensures only valid labels are produced.")
else:
    print(f"\n⚠️  WARNING: {failed} test(s) failed!")
    print("Constrained generation may have implementation issues.")

print("="*80)
