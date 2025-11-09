#!/usr/bin/env python3
"""
Comprehensive Zero-Shot Classification Test
Tests all example prompts with all label sets and records timing + results
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# API endpoint
API_URL = "http://localhost:5000/api/zero-shot/classify"

# Test examples
EXAMPLES = [
    {
        "name": "Positive review",
        "text": "This product exceeded all my expectations! The quality is outstanding and delivery was incredibly fast."
    },
    {
        "name": "Negative review",
        "text": "I'm very disappointed with this purchase. The item arrived damaged and customer service was unhelpful."
    },
    {
        "name": "Neutral review",
        "text": "The product works as described. Nothing particularly special, but it gets the job done."
    },
    {
        "name": "Mixed sentiment",
        "text": "I have mixed feelings about this. Some features are great, but others need improvement."
    }
]

# Label presets
LABEL_PRESETS = {
    "sentiment": ["positive", "negative", "neutral"],
    "intent": ["question", "complaint", "praise", "request", "information"],
    "urgency": ["urgent", "normal", "low-priority"],
    "cargo": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
}

# Model to test
MODEL = "SmolLM2 1.7B"

def classify_text(text: str, labels: List[str], use_logprobs: bool = True) -> Dict[str, Any]:
    """Send classification request and parse SSE stream"""
    
    payload = {
        "text": text,
        "candidate_labels": labels,
        "model": MODEL,
        "use_logprobs": use_logprobs,
        "abstain_threshold": None,
        "hypothesis_template": "This text is about {label}."
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        result = None
        processing_time = None
        
        # Parse SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                try:
                    data = json.loads(data_str)
                    
                    if 'done' in data and data['done']:
                        result = data.get('result')
                        processing_time = data.get('processing_time')
                        break
                    
                    if 'error' in data:
                        return {
                            'success': False,
                            'error': data['error'],
                            'elapsed_time': time.time() - start_time
                        }
                        
                except json.JSONDecodeError:
                    continue
        
        elapsed_time = time.time() - start_time
        
        if result:
            return {
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'elapsed_time': elapsed_time
            }
        else:
            return {
                'success': False,
                'error': 'No result received',
                'elapsed_time': elapsed_time
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def format_result(result: Dict[str, Any], example_name: str, preset_name: str) -> str:
    """Format result for display"""
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Example: {example_name}")
    lines.append(f"Label Set: {preset_name.upper()}")
    lines.append(f"{'='*80}")
    
    if not result['success']:
        lines.append(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        lines.append(f"Elapsed Time: {result['elapsed_time']:.2f}s")
        return '\n'.join(lines)
    
    res = result['result']
    
    # Top prediction
    lines.append(f"\n🏆 Top Prediction: {res['top_label'].upper()} ({res['top_score']*100:.1f}%)")
    
    # Timing
    lines.append(f"\n⏱️  Timing:")
    lines.append(f"  - Processing Time: {result['processing_time']:.2f}s")
    lines.append(f"  - Total Elapsed: {result['elapsed_time']:.2f}s")
    
    # All labels with scores
    lines.append(f"\n📊 All Labels:")
    for label_data in res['labels']:
        logprob_str = f" (logprob: {label_data['logprob']:.3f})" if label_data.get('logprob') is not None else ""
        lines.append(f"  - {label_data['label']:<25} {label_data['score']*100:>6.1f}%{logprob_str}")
    
    # Abstain indicator
    if res.get('should_abstain'):
        lines.append(f"\n⚠️  Low Confidence: Model suggests abstaining")
    
    return '\n'.join(lines)


def main():
    """Run comprehensive tests"""
    
    print("="*80)
    print("COMPREHENSIVE ZERO-SHOT CLASSIFICATION TEST")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Tests: {len(EXAMPLES)} examples × {len(LABEL_PRESETS)} label sets = {len(EXAMPLES) * len(LABEL_PRESETS)} tests")
    print("="*80)
    
    all_results = []
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'total_time': 0,
        'by_preset': {}
    }
    
    # Run all tests
    for example in EXAMPLES:
        for preset_name, labels in LABEL_PRESETS.items():
            stats['total'] += 1
            
            print(f"\nTesting: {example['name']} × {preset_name}...", end=' ', flush=True)
            
            result = classify_text(example['text'], labels)
            
            if result['success']:
                stats['success'] += 1
                stats['total_time'] += result['processing_time']
                print("✓")
            else:
                stats['failed'] += 1
                print("✗")
            
            # Track by preset
            if preset_name not in stats['by_preset']:
                stats['by_preset'][preset_name] = {
                    'count': 0,
                    'total_time': 0,
                    'success': 0
                }
            
            stats['by_preset'][preset_name]['count'] += 1
            if result['success']:
                stats['by_preset'][preset_name]['success'] += 1
                stats['by_preset'][preset_name]['total_time'] += result['processing_time']
            
            all_results.append({
                'example': example['name'],
                'preset': preset_name,
                'labels': labels,
                'text': example['text'],
                'result': result
            })
    
    # Print all results
    print("\n\n")
    print("="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for item in all_results:
        output = format_result(item['result'], item['example'], item['preset'])
        print(output)
    
    # Print summary statistics
    print("\n\n")
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal Tests: {stats['total']}")
    print(f"✓ Successful: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"✗ Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    
    if stats['success'] > 0:
        print(f"\nAverage Processing Time: {stats['total_time']/stats['success']:.2f}s")
        print(f"Total Processing Time: {stats['total_time']:.2f}s")
    
    print(f"\nBy Label Set:")
    for preset_name, preset_stats in stats['by_preset'].items():
        success_rate = preset_stats['success'] / preset_stats['count'] * 100
        avg_time = preset_stats['total_time'] / preset_stats['success'] if preset_stats['success'] > 0 else 0
        print(f"  {preset_name:10s}: {preset_stats['success']}/{preset_stats['count']} tests ({success_rate:>5.1f}%), avg {avg_time:.2f}s")
    
    # Save results to JSON
    output_file = f"zero_shot_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'model': MODEL,
                'timestamp': datetime.now().isoformat(),
                'total_tests': stats['total']
            },
            'statistics': stats,
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
