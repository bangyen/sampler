#!/usr/bin/env python3
"""
Debug logprobs behavior - examine what's happening with and without logprobs
"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"

# Single test case - refrigerated cargo
TEST_TEXT = "This container needs refrigeration for the perishable goods"
TEST_LABELS = ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
MODEL = "Qwen 2.5 7B"

def test_classification(use_logprobs: bool, verbose: bool = True):
    """Test classification and show detailed results"""
    
    payload = {
        "text": TEST_TEXT,
        "candidate_labels": TEST_LABELS,
        "model": MODEL,
        "use_logprobs": use_logprobs,
        "hypothesis_template": "This shipment is {label}."
    }
    
    print(f"\n{'='*80}")
    print(f"Testing: use_logprobs={use_logprobs}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        result = None
        processing_time = None
        
        # Parse SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    
                    # Show progress messages
                    if 'progress' in data and verbose:
                        print(f"  [PROGRESS] {data.get('message', '')}")
                    
                    if 'done' in data and data['done']:
                        result = data.get('result')
                        processing_time = data.get('processing_time')
                        break
                    
                    if 'error' in data:
                        print(f"  [ERROR] {data['error']}")
                        return None
                        
                except json.JSONDecodeError:
                    continue
        
        elapsed_time = time.time() - start_time
        
        if result:
            print(f"\n  ✓ Success!")
            print(f"  Processing Time: {processing_time:.2f}s")
            print(f"  Total Elapsed: {elapsed_time:.2f}s")
            print(f"\n  Top Prediction: {result['top_label']} ({result['top_score']*100:.1f}%)")
            print(f"  Expected: refrigerated cargo")
            print(f"  Correct: {result['top_label'] == 'refrigerated cargo'}")
            
            print(f"\n  All Labels:")
            for label_data in result['labels']:
                marker = "★" if label_data['label'] == result['top_label'] else " "
                logprob_str = f" (logprob: {label_data['logprob']:.3f})" if label_data.get('logprob') is not None else " (no logprob)"
                print(f"  {marker} {label_data['label']:<25s} {label_data['score']*100:>6.1f}%{logprob_str}")
            
            return result
        else:
            print(f"  ✗ No result received")
            return None
            
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


print("="*80)
print("DEBUGGING LOGPROBS MODE")
print("="*80)
print(f"Model: {MODEL}")
print(f"Text: '{TEST_TEXT}'")
print(f"Labels: {TEST_LABELS}")
print(f"Expected: refrigerated cargo")

# Test WITHOUT logprobs (fast mode)
result_fast = test_classification(use_logprobs=False)

time.sleep(1)

# Test WITH logprobs (slow mode)
result_slow = test_classification(use_logprobs=True)

# Compare
print(f"\n\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
if result_fast and result_slow:
    print(f"\nFAST MODE (use_logprobs=False):")
    print(f"  Predicted: {result_fast['top_label']}")
    print(f"  Confidence: {result_fast['top_score']*100:.1f}%")
    print(f"  Correct: {result_fast['top_label'] == 'refrigerated cargo'}")
    
    print(f"\nSLOW MODE (use_logprobs=True):")
    print(f"  Predicted: {result_slow['top_label']}")
    print(f"  Confidence: {result_slow['top_score']*100:.1f}%")
    print(f"  Correct: {result_slow['top_label'] == 'refrigerated cargo'}")
    
    print(f"\n🔍 Analysis:")
    if result_fast['top_label'] != result_slow['top_label']:
        print(f"  ⚠️  Different predictions!")
        print(f"  Fast mode: {result_fast['top_label']}")
        print(f"  Slow mode: {result_slow['top_label']}")
    else:
        print(f"  ✓ Same prediction: {result_fast['top_label']}")
print("="*80)
