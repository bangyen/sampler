#!/usr/bin/env python3
"""
Measure NER model memory more accurately
"""
import gc
import psutil
import os


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def measure_ner(model_id):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    gc.collect()
    baseline = get_memory_mb()

    print(f"\n{model_id}:")
    print(f"  Baseline: {baseline:.1f} MB")

    # Load full model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    after_load = get_memory_mb()
    memory_used = after_load - baseline

    print(f"  After load: {after_load:.1f} MB")
    print(f"  Memory used: {memory_used:.1f} MB ({memory_used/1024:.2f} GB)")

    del model, tokenizer
    gc.collect()

    return memory_used / 1024


print("NER MODEL MEMORY MEASUREMENTS")
print("=" * 50)

results = {}
results["BERT Base NER"] = measure_ner("dslim/bert-base-NER")
results["BERT Large NER"] = measure_ner("dslim/bert-large-NER")
results["RoBERTa Large NER"] = measure_ner("Jean-Baptiste/roberta-large-ner-english")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
for name, mem in results.items():
    print(f"{name:25s}: {mem:.2f} GB")
