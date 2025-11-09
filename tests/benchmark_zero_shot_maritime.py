#!/usr/bin/env python3
"""
Maritime Zero-Shot Classification Benchmark
Compares accuracy and speed across all available models for maritime cargo classification
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

API_URL = "http://localhost:5000/api/zero-shot/classify"
MODELS_URL = "http://localhost:5000/api/models"

MARITIME_EXAMPLES = [
    {
        "text": "This container needs refrigeration for the perishable goods",
        "expected": "refrigerated cargo",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
    },
    {
        "text": "The shipment contains chemicals that require special handling and safety precautions",
        "expected": "hazardous materials",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
    },
    {
        "text": "Standard palletized goods, no special requirements",
        "expected": "standard cargo",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
    },
    {
        "text": "Heavy machinery exceeding standard container dimensions",
        "expected": "oversized freight",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
    },
    {
        "text": "Frozen meat products maintained at -18°C during transport",
        "expected": "refrigerated cargo",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
    },
    {
        "text": "Flammable liquids with UN hazard classification",
        "expected": "hazardous materials",
        "labels": ["standard cargo", "refrigerated cargo", "hazardous materials", "oversized freight"]
    }
]

@dataclass
class BenchmarkResult:
    model: str
    use_logprobs: bool
    text: str
    expected_label: str
    predicted_label: str
    confidence: float
    is_correct: bool
    processing_time: float
    elapsed_time: float
    error: Optional[str] = None

def get_available_models() -> List[str]:
    """Fetch list of available models from the API"""
    try:
        response = requests.get(MODELS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = list(data.get("models", {}).keys())
        print(f"Available models: {models}")
        return models
    except Exception as e:
        print(f"Warning: Failed to fetch models from API: {e}")
        print("Using default model list...")
        return ["SmolLM2 1.7B", "Qwen 2.5 0.5B", "Qwen 2.5 7B", "BitNet b1.58 2B"]

def classify_text(text: str, labels: List[str], model: str, use_logprobs: bool) -> Dict[str, Any]:
    """Send classification request and parse SSE stream"""
    
    payload = {
        "text": text,
        "candidate_labels": labels,
        "model": model,
        "use_logprobs": use_logprobs,
        "hypothesis_template": "This shipment is {label}."
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

def run_benchmark(models: List[str]) -> List[BenchmarkResult]:
    """Run benchmark across all models and test cases"""
    
    results = []
    total_tests = len(models) * len(MARITIME_EXAMPLES) * 2  # 2 modes: with/without logprobs
    current_test = 0
    
    print(f"\n{'='*80}")
    print(f"Starting benchmark: {len(models)} models × {len(MARITIME_EXAMPLES)} examples × 2 modes = {total_tests} tests")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 80)
        
        for use_logprobs in [True, False]:
            mode = "WITH logprobs" if use_logprobs else "WITHOUT logprobs"
            print(f"\n  Mode: {mode}")
            
            for i, example in enumerate(MARITIME_EXAMPLES, 1):
                current_test += 1
                print(f"    [{current_test}/{total_tests}] Example {i}: '{example['text'][:50]}...'", end=' ')
                
                response = classify_text(
                    text=example['text'],
                    labels=example['labels'],
                    model=model,
                    use_logprobs=use_logprobs
                )
                
                if response['success']:
                    result = response['result']
                    predicted = result['top_label']
                    confidence = result['top_score']
                    is_correct = predicted == example['expected']
                    
                    benchmark_result = BenchmarkResult(
                        model=model,
                        use_logprobs=use_logprobs,
                        text=example['text'],
                        expected_label=example['expected'],
                        predicted_label=predicted,
                        confidence=confidence,
                        is_correct=is_correct,
                        processing_time=response['processing_time'],
                        elapsed_time=response['elapsed_time']
                    )
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} {predicted} ({confidence*100:.1f}%) - {response['processing_time']:.2f}s")
                else:
                    benchmark_result = BenchmarkResult(
                        model=model,
                        use_logprobs=use_logprobs,
                        text=example['text'],
                        expected_label=example['expected'],
                        predicted_label="",
                        confidence=0.0,
                        is_correct=False,
                        processing_time=0.0,
                        elapsed_time=response['elapsed_time'],
                        error=response.get('error', 'Unknown error')
                    )
                    print(f"✗ Error: {response.get('error')}")
                
                results.append(benchmark_result)
                time.sleep(0.3)  # Small delay between requests
    
    return results

def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze benchmark results and compute statistics"""
    
    analysis = {
        'by_model': {},
        'by_model_mode': {}
    }
    
    for result in results:
        if result.error:
            continue
        
        # By model (aggregated across both modes)
        if result.model not in analysis['by_model']:
            analysis['by_model'][result.model] = {
                'total': 0,
                'correct': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        stats = analysis['by_model'][result.model]
        stats['total'] += 1
        stats['correct'] += 1 if result.is_correct else 0
        stats['total_time'] += result.processing_time
        stats['min_time'] = min(stats['min_time'], result.processing_time)
        stats['max_time'] = max(stats['max_time'], result.processing_time)
        
        # By model and mode
        key = f"{result.model} ({'logprobs' if result.use_logprobs else 'fast'})"
        if key not in analysis['by_model_mode']:
            analysis['by_model_mode'][key] = {
                'model': result.model,
                'use_logprobs': result.use_logprobs,
                'total': 0,
                'correct': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        mode_stats = analysis['by_model_mode'][key]
        mode_stats['total'] += 1
        mode_stats['correct'] += 1 if result.is_correct else 0
        mode_stats['total_time'] += result.processing_time
        mode_stats['min_time'] = min(mode_stats['min_time'], result.processing_time)
        mode_stats['max_time'] = max(mode_stats['max_time'], result.processing_time)
    
    # Compute derived metrics
    for stats in analysis['by_model'].values():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        stats['avg_time'] = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
    
    for stats in analysis['by_model_mode'].values():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        stats['avg_time'] = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
    
    return analysis

def print_report(analysis: Dict[str, Any]):
    """Print formatted comparison report"""
    
    print("\n\n")
    print("="*80)
    print("MARITIME BENCHMARK RESULTS - MODEL COMPARISON")
    print("="*80)
    
    # Overall model comparison
    print("\n📊 OVERALL MODEL PERFORMANCE (both modes combined)")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Avg Time':>12} {'Min Time':>12} {'Max Time':>12}")
    print("-" * 80)
    
    sorted_models = sorted(
        analysis['by_model'].items(),
        key=lambda x: (-x[1]['accuracy'], x[1]['avg_time'])  # Sort by accuracy desc, then speed asc
    )
    
    for model, stats in sorted_models:
        print(f"{model:<25} {stats['accuracy']*100:>9.1f}% {stats['avg_time']:>11.2f}s {stats['min_time']:>11.2f}s {stats['max_time']:>11.2f}s")
    
    # Detailed by mode
    print("\n\n📊 DETAILED PERFORMANCE BY MODE")
    print("-" * 80)
    print(f"{'Model + Mode':<40} {'Accuracy':>10} {'Avg Time':>12} {'Tests':>8}")
    print("-" * 80)
    
    sorted_modes = sorted(
        analysis['by_model_mode'].items(),
        key=lambda x: (-x[1]['accuracy'], x[1]['avg_time'])
    )
    
    for key, stats in sorted_modes:
        print(f"{key:<40} {stats['accuracy']*100:>9.1f}% {stats['avg_time']:>11.2f}s {stats['total']:>8}")
    
    # Find best options
    print("\n\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    if sorted_models:
        best_accuracy = sorted_models[0]
        fastest = min(analysis['by_model'].items(), key=lambda x: x[1]['avg_time'])
        
        print(f"\n✅ Most Accurate: {best_accuracy[0]}")
        print(f"   Accuracy: {best_accuracy[1]['accuracy']*100:.1f}%")
        print(f"   Avg Time: {best_accuracy[1]['avg_time']:.2f}s")
        
        print(f"\n⚡ Fastest: {fastest[0]}")
        print(f"   Accuracy: {fastest[1]['accuracy']*100:.1f}%")
        print(f"   Avg Time: {fastest[1]['avg_time']:.2f}s")
        
        # Speed comparison
        if best_accuracy[0] != fastest[0]:
            speedup = best_accuracy[1]['avg_time'] / fastest[1]['avg_time']
            accuracy_diff = (best_accuracy[1]['accuracy'] - fastest[1]['accuracy']) * 100
            
            print(f"\n🔍 Trade-off Analysis:")
            print(f"   {best_accuracy[0]} is {speedup:.1f}x slower than {fastest[0]}")
            print(f"   but provides {accuracy_diff:+.1f}% better accuracy")
            
            if speedup > 3 and accuracy_diff < 10:
                print(f"\n   💭 Consider: The faster model ({fastest[0]}) may be sufficient")
                print(f"      unless the {accuracy_diff:.1f}% accuracy gain is critical")
            elif accuracy_diff > 15:
                print(f"\n   💭 Consider: The accuracy improvement ({accuracy_diff:.1f}%) may justify")
                print(f"      the {speedup:.1f}x speed penalty for production use")
    
    print("\n" + "="*80)

def main():
    """Run the maritime zero-shot classification benchmark"""
    
    print("="*80)
    print("MARITIME ZERO-SHOT CLASSIFICATION BENCHMARK")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Cases: {len(MARITIME_EXAMPLES)} maritime cargo examples")
    
    # Get available models
    models = get_available_models()
    
    if not models:
        print("Error: No models available")
        return
    
    # Run benchmark
    results = run_benchmark(models)
    
    # Analyze and report
    analysis = analyze_results(results)
    print_report(analysis)
    
    # Save results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"maritime_benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_cases': len(MARITIME_EXAMPLES),
                'models': models,
                'total_tests': len(results)
            },
            'results': [asdict(r) for r in results],
            'analysis': analysis
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
