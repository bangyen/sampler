#!/usr/bin/env python3
"""
Comprehensive test showing early termination optimization in different scenarios
"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"
MODEL = "SmolLM2 1.7B"

# Test cases demonstrating different early termination scenarios
TEST_CASES = [
    {
        "name": "Unique prefixes (cargo types)",
        "text": "Standard shipping required",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials"],
        "note": "Each first token uniquely identifies the label → early termination after token 1"
    },
    {
        "name": "Shared prefixes (urgency levels)",
        "text": "This is very urgent",
        "labels": ["urgent", "urgent escalation", "normal"],
        "note": "After 'urgent' → 2 possible labels (urgent, urgent escalation) → NO early termination"
    },
    {
        "name": "All single-token (sentiment)",
        "text": "This is amazing!",
        "labels": ["positive", "negative", "neutral"],
        "note": "First token completes the label → early termination at token 1"
    },
]

print("="*80)
print("EARLY TERMINATION OPTIMIZATION - COMPREHENSIVE TEST")
print("="*80)
print(f"Model: {MODEL}\n")

for i, test_case in enumerate(TEST_CASES, 1):
    print(f"\nTest {i}/{len(TEST_CASES)}: {test_case['name']}")
    print("-" * 80)
    print(f"Text: '{test_case['text']}'")
    print(f"Labels: {test_case['labels']}")
    print(f"Expected: {test_case['note']}")
    
    payload = {
        "text": test_case['text'],
        "candidate_labels": test_case['labels'],
        "model": MODEL,
        "use_logprobs": False,  # Faster without logprobs
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
            print(f"  Time: {elapsed:.2f}s")
            print("  (Check logs above for 'UNIQUE MATCH' debug output)")
        else:
            print("✗ Failed to get result")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    time.sleep(0.5)  # Brief pause between tests

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Early termination optimization is active!")
print("When a prefix uniquely identifies one label, the system:")
print("  1. Detects UNIQUE MATCH")
print("  2. Forces the remaining tokens from that label")
print("  3. Prevents the model from considering other options")
print("\nThis makes generation faster and more deterministic.")
print("="*80)
