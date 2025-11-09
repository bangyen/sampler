#!/usr/bin/env python3
"""
Compare performance and confidence scores between logprobs enabled/disabled modes
"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"
MODEL = "SmolLM2 1.7B"

# Test cases
TEST_CASES = [
    {
        "name": "Sentiment Analysis",
        "text": "This is amazing!",
        "labels": ["positive", "negative", "neutral"],
    },
    {
        "name": "Cargo Classification",
        "text": "This container needs refrigeration",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"],
    },
    {
        "name": "Urgency Detection",
        "text": "This is very urgent and requires immediate attention",
        "labels": ["urgent", "urgent escalation", "normal"],
    },
    {
        "name": "Intent Classification",
        "text": "I have a question about shipping",
        "labels": ["question", "complaint", "praise", "request", "information"],
    }
]

def test_classification(text: str, labels: list, use_logprobs: bool) -> dict:
    """Test classification and return result with timing"""
    
    payload = {
        "text": text,
        "candidate_labels": labels,
        "model": MODEL,
        "use_logprobs": use_logprobs,
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
                    if 'error' in data:
                        return {'success': False, 'error': data['error']}
                except json.JSONDecodeError:
                    continue
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'result': result,
            'elapsed_time': elapsed
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


print("="*80)
print("LOGPROBS TIMING COMPARISON TEST")
print("="*80)
print(f"Model: {MODEL}")
print(f"Purpose: Compare speed and confidence quality with/without logprobs\n")

all_results = []
total_tests = len(TEST_CASES)

for i, test_case in enumerate(TEST_CASES, 1):
    print(f"\nTest {i}/{total_tests}: {test_case['name']}")
    print("-" * 80)
    print(f"Text: '{test_case['text']}'")
    print(f"Labels: {test_case['labels']}")
    
    # Run WITHOUT logprobs (fast mode)
    print("\n  [1] WITHOUT logprobs (Fast - uniform scores):")
    result_fast = test_classification(test_case['text'], test_case['labels'], use_logprobs=False)
    
    if result_fast['success']:
        res = result_fast['result']
        print(f"      Label: {res['top_label']}")
        print(f"      Confidence: {res['top_score']*100:.1f}% (uniform)")
        print(f"      Time: {result_fast['elapsed_time']:.2f}s")
        fast_time = result_fast['elapsed_time']
    else:
        print(f"      ERROR: {result_fast.get('error')}")
        fast_time = None
    
    time.sleep(0.5)  # Brief pause between requests
    
    # Run WITH logprobs (accurate mode)
    print("\n  [2] WITH logprobs (Accurate - real confidence scores):")
    result_accurate = test_classification(test_case['text'], test_case['labels'], use_logprobs=True)
    
    if result_accurate['success']:
        res = result_accurate['result']
        print(f"      Label: {res['top_label']}")
        print(f"      Confidence: {res['top_score']*100:.1f}% (from model logprobs)")
        print(f"      Time: {result_accurate['elapsed_time']:.2f}s")
        accurate_time = result_accurate['elapsed_time']
        
        # Show all label scores
        print(f"      All scores:")
        for label_obj in res['labels'][:3]:  # Top 3
            print(f"        - {label_obj['label']}: {label_obj['score']*100:.1f}%")
    else:
        print(f"      ERROR: {result_accurate.get('error')}")
        accurate_time = None
    
    # Calculate speedup
    if fast_time and accurate_time:
        speedup = accurate_time / fast_time
        print(f"\n  ⚡ Speedup: {speedup:.2f}x faster without logprobs")
        all_results.append({
            'test': test_case['name'],
            'fast_time': fast_time,
            'accurate_time': accurate_time,
            'speedup': speedup
        })
    
    time.sleep(0.5)  # Brief pause between tests

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all_results:
    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"\nDetailed results:")
    for r in all_results:
        print(f"  {r['test']}:")
        print(f"    Without logprobs: {r['fast_time']:.2f}s")
        print(f"    With logprobs:    {r['accurate_time']:.2f}s")
        print(f"    Speedup:          {r['speedup']:.2f}x")

print("\n" + "="*80)
print("TRADEOFFS")
print("="*80)
print("WITHOUT logprobs (Fast):")
print("  ✓ ~2x faster processing")
print("  ✓ Perfect for: Label-only classification")
print("  ✗ Uniform confidence scores (not useful)")
print("\nWITH logprobs (Accurate):")
print("  ✓ Real confidence scores from model probabilities")
print("  ✓ Perfect for: Confidence-based decisions (e.g., 'only act if >80%')")
print("  ✗ ~2x slower processing")
print("="*80)
