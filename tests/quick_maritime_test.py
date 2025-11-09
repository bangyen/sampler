#!/usr/bin/env python3
"""Quick test to verify logprobs fix with maritime examples"""

import requests
import json
import time

API_URL = "http://localhost:5000/api/zero-shot/classify"

TESTS = [
    {
        "text": "This container needs refrigeration for the perishable goods",
        "expected": "refrigerated cargo"
    },
    {
        "text": "The shipment contains chemicals that require special handling",
        "expected": "hazardous materials"
    },
    {
        "text": "Standard palletized goods, no special requirements",
        "expected": "standard cargo"
    }
]

LABELS = ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
MODEL = "Qwen 2.5 7B"

def test(use_logprobs):
    mode = "WITH logprobs" if use_logprobs else "WITHOUT logprobs"
    print(f"\n{mode}:")
    print("-" * 60)
    
    correct = 0
    for i, test in enumerate(TESTS, 1):
        payload = {
            "text": test['text'],
            "candidate_labels": LABELS,
            "model": MODEL,
            "use_logprobs": use_logprobs,
            "hypothesis_template": "This shipment is {label}."
        }
        
        try:
            response = requests.post(API_URL, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if 'done' in data and data['done']:
                        result = data['result']
                        predicted = result['top_label']
                        is_correct = predicted == test['expected']
                        correct += is_correct
                        
                        status = "✓" if is_correct else "✗"
                        print(f"  {status} Test {i}: {predicted} (expected: {test['expected']})")
                        break
        except Exception as e:
            print(f"  ✗ Test {i}: Error - {e}")
        
        time.sleep(0.5)
    
    accuracy = (correct / len(TESTS)) * 100
    print(f"\n  Accuracy: {correct}/{len(TESTS)} ({accuracy:.1f}%)")
    return accuracy

print("="*60)
print("MARITIME CLASSIFICATION - LOGPROBS FIX VERIFICATION")
print("="*60)
print(f"Model: {MODEL}")
print(f"Tests: {len(TESTS)}")

fast_acc = test(use_logprobs=False)
slow_acc = test(use_logprobs=True)

print(f"\n{'='*60}")
print("SUMMARY")
print("="*60)
print(f"Fast mode (use_logprobs=False): {fast_acc:.1f}%")
print(f"Slow mode (use_logprobs=True):  {slow_acc:.1f}%")

if slow_acc == fast_acc:
    print(f"\n✓ FIX VERIFIED: Both modes now have equal accuracy!")
else:
    print(f"\n⚠️  Modes still differ: {abs(slow_acc - fast_acc):.1f}% difference")

print("="*60)
