#!/usr/bin/env python3
"""
Test speed difference between logprob scoring vs fast mode
"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"
TEST_TEXT = "This product exceeded all my expectations! The quality is outstanding."
TEST_LABELS = ["positive", "negative", "neutral"]
MODEL = "SmolLM2 1.7B"

def test_classification(use_logprobs: bool) -> dict:
    """Test classification with or without logprobs"""
    
    payload = {
        "text": TEST_TEXT,
        "candidate_labels": TEST_LABELS,
        "model": MODEL,
        "use_logprobs": use_logprobs,
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
print("SPEED COMPARISON: Logprob Scoring vs Fast Mode")
print("="*80)
print(f"Test: '{TEST_TEXT}'")
print(f"Labels: {TEST_LABELS}")
print(f"Model: {MODEL}\n")

# Test WITH logprobs (accurate confidence)
print("Test 1: WITH Logprob Scoring (Accurate Confidence)")
print("-" * 80)
result_with = test_classification(use_logprobs=True)

if result_with['success']:
    res = result_with['result']
    print(f"✓ Processing Time: {result_with['processing_time']:.2f}s")
    print(f"  Top Label: {res['top_label']} ({res['top_score']*100:.1f}%)")
    print(f"  All Labels:")
    for label_data in res['labels']:
        logprob_str = f" (logprob: {label_data['logprob']:.3f})" if label_data.get('logprob') is not None else ""
        print(f"    - {label_data['label']:<10s} {label_data['score']*100:>6.1f}%{logprob_str}")
else:
    print(f"✗ Failed: {result_with.get('error')}")

print()

# Test WITHOUT logprobs (fast mode)
print("Test 2: WITHOUT Logprob Scoring (Fast Mode)")
print("-" * 80)
result_without = test_classification(use_logprobs=False)

if result_without['success']:
    res = result_without['result']
    print(f"✓ Processing Time: {result_without['processing_time']:.2f}s")
    print(f"  Top Label: {res['top_label']} ({res['top_score']*100:.1f}%)")
    print(f"  All Labels:")
    for label_data in res['labels']:
        print(f"    - {label_data['label']:<10s} {label_data['score']*100:>6.1f}%")
else:
    print(f"✗ Failed: {result_without.get('error')}")

print()
print("="*80)
print("COMPARISON")
print("="*80)

if result_with['success'] and result_without['success']:
    time_with = result_with['processing_time']
    time_without = result_without['processing_time']
    speedup = time_with / time_without
    time_saved = time_with - time_without
    
    print(f"WITH Logprobs:    {time_with:.2f}s (Accurate confidence scores)")
    print(f"WITHOUT Logprobs: {time_without:.2f}s (Uniform/fast scores)")
    print(f"\n⚡ Speedup: {speedup:.1f}x faster")
    print(f"⏱️  Time Saved: {time_saved:.2f}s ({time_saved/time_with*100:.1f}%)")
    print(f"\n💡 Use logprobs=False when you only need the label (not confidence)")
    print(f"💡 Use logprobs=True when you need accurate confidence scores")

print("="*80)
