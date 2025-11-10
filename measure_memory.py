#!/usr/bin/env python3
"""
Measure actual memory usage of models
"""
import gc
import psutil
import os
import sys

def get_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def measure_model(name, load_fn):
    """Measure memory usage of a model"""
    # Clear memory
    gc.collect()
    
    # Get baseline
    baseline = get_memory_mb()
    print(f"\n{name}:")
    print(f"  Baseline: {baseline:.1f} MB")
    
    try:
        # Load model
        print(f"  Loading...")
        model_obj = load_fn()
        
        # Measure after loading
        after_load = get_memory_mb()
        memory_used = after_load - baseline
        
        print(f"  After load: {after_load:.1f} MB")
        print(f"  Memory used: {memory_used:.1f} MB ({memory_used/1024:.2f} GB)")
        
        # Clean up
        del model_obj
        gc.collect()
        
        return memory_used / 1024  # Return GB
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

# Qwen 0.5B FP16
def load_qwen_05b():
    from transformers import AutoModelForCausalLM
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to("cpu")
    return model

# SmolLM2 1.7B GGUF
def load_smollm_gguf():
    from inference.bitnet_inference import BitNetInference, download_gguf_model
    model_path = download_gguf_model(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "smollm2-1.7b-instruct-q4_k_m.gguf"
    )
    inference = BitNetInference(model_path, n_ctx=2048)
    return inference

# SmolLM2 1.7B FP16 (fallback)
def load_smollm_fp16():
    from transformers import AutoModelForCausalLM
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to("cpu")
    return model

# Qwen 7B GGUF
def load_qwen_7b_gguf():
    from inference.bitnet_inference import BitNetInference, download_gguf_model
    model_path = download_gguf_model(
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )
    inference = BitNetInference(model_path, n_ctx=2048)
    return inference

# BitNet 2B
def load_bitnet():
    from inference.bitnet_cpp_bridge import load_bitnet_model
    bridge = load_bitnet_model(
        "microsoft/bitnet-b1.58-2B-4T-gguf",
        "ggml-model-i2_s.gguf"
    )
    return bridge

# BERT Base NER
def load_bert_base_ner():
    from transformers import pipeline
    nlp = pipeline("ner", model="dslim/bert-base-NER", device=-1)
    return nlp

# BERT Large NER
def load_bert_large_ner():
    from transformers import pipeline
    nlp = pipeline("ner", model="dslim/bert-large-NER", device=-1)
    return nlp

# RoBERTa Large NER
def load_roberta_large_ner():
    from transformers import pipeline
    nlp = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", device=-1)
    return nlp

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MEMORY MEASUREMENT")
    print("=" * 60)
    
    results = {}
    
    # Measure each model
    print("\n### CLASSIFICATION MODELS ###")
    results["Qwen 0.5B FP16"] = measure_model("Qwen 0.5B (FP16)", load_qwen_05b)
    
    try:
        results["SmolLM2 1.7B GGUF"] = measure_model("SmolLM2 1.7B (GGUF)", load_smollm_gguf)
    except:
        results["SmolLM2 1.7B FP16"] = measure_model("SmolLM2 1.7B (FP16)", load_smollm_fp16)
    
    try:
        results["Qwen 7B GGUF"] = measure_model("Qwen 7B (GGUF)", load_qwen_7b_gguf)
    except Exception as e:
        print(f"Skipping Qwen 7B: {e}")
    
    try:
        results["BitNet 2B"] = measure_model("BitNet 2B", load_bitnet)
    except Exception as e:
        print(f"Skipping BitNet: {e}")
    
    print("\n### NER MODELS ###")
    results["BERT Base NER"] = measure_model("BERT Base NER", load_bert_base_ner)
    results["BERT Large NER"] = measure_model("BERT Large NER", load_bert_large_ner)
    results["RoBERTa Large NER"] = measure_model("RoBERTa Large NER", load_roberta_large_ner)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, mem in results.items():
        if mem is not None:
            print(f"{name:30s}: {mem:.2f} GB")
        else:
            print(f"{name:30s}: FAILED")
