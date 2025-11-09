#!/usr/bin/env python3
"""Simple test to examine logprobs with detailed logging"""

import requests
import json

API_URL = "http://localhost:5000/api/zero-shot/classify"

payload = {
    "text": "This container needs refrigeration for the perishable goods",
    "candidate_labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"],
    "model": "Qwen 2.5 7B",
    "use_logprobs": True,
    "hypothesis_template": "This shipment is {label}."
}

print("Testing with logprobs=True to see debug output...")
print("Check the workflow logs for [DEBUG] messages\n")

response = requests.post(API_URL, json=payload, stream=True, timeout=60)

for line in response.iter_lines(decode_unicode=True):
    if line.startswith('data: '):
        data = json.loads(line[6:])
        if 'done' in data and data['done']:
            result = data['result']
            print(f"Result: {result['top_label']} ({result['top_score']*100:.1f}%)")
            break

print("\nNow check the workflow logs for detailed DEBUG output")
