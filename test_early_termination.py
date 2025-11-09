#!/usr/bin/env python3
"""
Test early termination optimization when prefix uniquely identifies one label
"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"
MODEL = "SmolLM2 1.7B"

# Test case where early termination should happen
test_text = "This container needs refrigeration"
test_labels = ["standard cargo", "refrigerated cargo", "hazardous materials"]

print("="*80)
print("EARLY TERMINATION OPTIMIZATION TEST")
print("="*80)
print(f"Text: '{test_text}'")
print(f"Labels: {test_labels}")
print("\nExpected behavior:")
print("- After 'standard' → UNIQUE MATCH (only 'standard cargo' possible) → force ' cargo'")
print("- After 'refrigerated' → UNIQUE MATCH (only 'refrigerated cargo' possible) → force ' cargo'")
print("- After 'hazardous' → UNIQUE MATCH (only 'hazardous materials' possible) → force ' materials'")
print("\nRunning classification...\n")

payload = {
    "text": test_text,
    "candidate_labels": test_labels,
    "model": MODEL,
    "use_logprobs": False,  # Disable for faster testing
}

start_time = time.time()

try:
    response = requests.post(API_URL, json=payload, stream=True, timeout=120)
    response.raise_for_status()
    
    result = None
    
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith('data: '):
            data_str = line[6:]
            try:
                data = json.loads(data_str)
                if 'done' in data and data['done']:
                    result = data.get('result')
                    break
            except json.JSONDecodeError:
                continue
    
    elapsed = time.time() - start_time
    
    if result:
        print(f"✓ Result: '{result['top_label']}'")
        print(f"✓ Processing time: {result['processing_time']:.2f}s")
        print(f"✓ Total time: {elapsed:.2f}s")
        print("\nCheck the workflow logs for debug output showing:")
        print("  [DEBUG] Current prefix: '...' → UNIQUE MATCH, forcing next token")
    else:
        print("✗ Failed to get result")
        
except Exception as e:
    print(f"✗ Error: {e}")

print("="*80)
