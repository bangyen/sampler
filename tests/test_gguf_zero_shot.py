#!/usr/bin/env python3
"""
Test GGUF backend for zero-shot classification
"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"

# Test with SmolLM2 GGUF model
TEST_CASES = [
    {
        "text": "This is amazing!",
        "labels": ["positive", "negative", "neutral"],
        "model": "SmolLM2 1.7B GGUF"  # GGUF model
    },
    {
        "text": "This container needs refrigeration",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials"],
        "model": "SmolLM2 1.7B GGUF"  # GGUF model
    }
]

print("="*80)
print("GGUF BACKEND ZERO-SHOT CLASSIFICATION TEST")
print("="*80)

for i, test_case in enumerate(TEST_CASES, 1):
    print(f"\nTest {i}/{len(TEST_CASES)}")
    print(f"Text: '{test_case['text']}'")
    print(f"Labels: {test_case['labels']}")
    print(f"Model: {test_case['model']}")
    
    # Test without logprobs (faster)
    print("\n  [1] WITHOUT logprobs (Fast):")
    payload = {
        "text": test_case['text'],
        "candidate_labels": test_case['labels'],
        "model": test_case['model'],
        "use_logprobs": False
    }
    
    start = time.time()
    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'done' in data and data['done']:
                    result = data.get('result')
                    elapsed = time.time() - start
                    print(f"      ✓ Label: {result['top_label']}")
                    print(f"      ✓ Time: {elapsed:.2f}s")
                    break
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    time.sleep(0.5)
    
    # Test with logprobs (slower but accurate)
    print("\n  [2] WITH logprobs (Accurate):")
    payload['use_logprobs'] = True
    
    start = time.time()
    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'done' in data and data['done']:
                    result = data.get('result')
                    elapsed = time.time() - start
                    print(f"      ✓ Label: {result['top_label']}")
                    print(f"      ✓ Confidence: {result['top_score']*100:.1f}%")
                    print(f"      ✓ Time: {elapsed:.2f}s")
                    break
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    time.sleep(0.5)

print("\n" + "="*80)
print("Check the workflow logs for:")
print("  [DEBUG] Using GGUF backend for zero-shot classification")
print("  [DEBUG] Constrained generation initialized:")
print("  [DEBUG] GGUF logprobs extracted:")
print("="*80)
