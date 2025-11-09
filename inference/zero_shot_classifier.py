import json
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from transformers.generation.stopping_criteria import StoppingCriteria


class JSONCompletionStoppingCriteria(StoppingCriteria):
    """Stop generation when valid JSON is detected"""
    
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.checked_cache = {}
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # Decode only the generated portion (skip prompt)
        generated_ids = input_ids[0][self.prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        num_tokens = len(generated_ids)
        
        # Quick check: must have at least opening and closing braces
        if not generated_text or '{' not in generated_text or '}' not in generated_text:
            return torch.tensor(False, dtype=torch.bool)
        
        # Check cache to avoid redundant parsing
        text_hash = hash(generated_text)
        if text_hash in self.checked_cache:
            return torch.tensor(self.checked_cache[text_hash], dtype=torch.bool)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(generated_text)
            # Verify it has the expected structure for zero-shot classification
            has_labels = 'labels' in parsed and isinstance(parsed['labels'], list) and len(parsed['labels']) > 0
            has_top = 'top_label' in parsed and 'top_score' in parsed
            
            is_complete = has_labels and has_top
            self.checked_cache[text_hash] = is_complete
            
            if is_complete:
                print(f"[EARLY STOP] Valid JSON detected at {num_tokens} tokens. Stopping generation.")
            
            return torch.tensor(is_complete, dtype=torch.bool)
        except (json.JSONDecodeError, ValueError):
            self.checked_cache[text_hash] = False
            return torch.tensor(False, dtype=torch.bool)


class ClassificationLabel(BaseModel):
    """Individual classification result for a label"""
    label: str = Field(..., description="The classification label")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized confidence score (0-1)")
    logprob: Optional[float] = Field(None, description="Log probability for calibration/confidence")


class ZeroShotResult(BaseModel):
    """Schema-locked JSON output for zero-shot classification"""
    text: str = Field(..., description="The input text that was classified")
    labels: List[ClassificationLabel] = Field(..., description="All classification results with scores")
    top_label: str = Field(..., description="The highest scoring label")
    top_score: float = Field(..., ge=0.0, le=1.0, description="The highest confidence score")
    should_abstain: bool = Field(False, description="Whether to abstain based on confidence threshold")
    abstain_threshold: Optional[float] = Field(None, description="Threshold used for abstain decision")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product!",
                "labels": [
                    {"label": "positive", "score": 0.95, "logprob": -0.051},
                    {"label": "negative", "score": 0.03, "logprob": -3.507},
                    {"label": "neutral", "score": 0.02, "logprob": -3.912}
                ],
                "top_label": "positive",
                "top_score": 0.95,
                "should_abstain": False,
                "abstain_threshold": 0.5
            }
        }


def build_label_only_prompt(
    text: str,
    candidate_labels: List[str]
) -> str:
    """
    Build a simple prompt that asks model to output only the label.
    For small models - avoids hallucinated confidence scores.
    """
    labels_str = ", ".join([f'"{label}"' for label in candidate_labels])
    
    prompt = f"""Classify the following text into one of these categories: {labels_str}

Text: {text}

Output only the category name that best matches the text (nothing else):"""
    
    return prompt


def build_zero_shot_prompt(
    text: str,
    candidate_labels: List[str],
    hypothesis_template: str = "This text is about {label}."
) -> str:
    """
    Build a prompt for zero-shot classification with JSON schema enforcement.
    DEPRECATED: Use build_label_only_prompt + logprob scoring instead.
    This is kept for backward compatibility with larger models.
    """
    
    labels_str = json.dumps(candidate_labels)
    
    # Build label placeholders without biased scores
    label_examples = []
    for label in candidate_labels:
        label_examples.append(f'    {{"label": "{label}", "score": <number between 0-1>}}')
    labels_format = ",\n".join(label_examples)
    
    prompt = f"""You are a text classifier. Analyze the text and assign confidence scores to each category based on the content.

Text to classify: {text}

Available categories: {labels_str}

Instructions:
1. Read the text carefully and evaluate which category fits best
2. Assign a confidence score (0.0 to 1.0) to each category - USE ONLY 1-2 DECIMAL PLACES (e.g., 0.9, 0.85, 0.1)
3. Higher scores mean better match, lower scores mean poor match
4. The category with the highest score is the top prediction
5. Output ONLY valid JSON with SHORT decimal numbers, no other text

Required JSON format (use SHORT decimals like 0.9, not 0.9999999):
{{
  "labels": [
{labels_format}
  ],
  "top_label": "<category with highest score>",
  "top_score": <highest score value>
}}

JSON:"""
    
    return prompt


def extract_logprobs_from_sequences(
    model,
    tokenizer,
    text: str,
    candidate_labels: List[str],
    hypothesis_template: str,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute the log probability of each candidate label sequence.
    Uses proper sequence probability: sum of log probs of generated label tokens.
    """
    logprobs = {}
    
    try:
        for label in candidate_labels:
            hypothesis = hypothesis_template.replace("{label}", label)
            prompt = f"Text: {text}\nClassification: {hypothesis}\nLabel:"
            
            full_sequence = f"{prompt} {label}"
            
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=480
            ).to(device)
            
            full_inputs = tokenizer(
                full_sequence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            prompt_len = prompt_inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model(**full_inputs, return_dict=True)
                logits = outputs.logits
                
                label_logprobs = []
                for i in range(prompt_len, full_inputs['input_ids'].shape[1]):
                    if i - 1 >= 0 and i - 1 < logits.shape[1]:
                        token_logits = logits[0, i - 1, :]
                        token_probs = F.softmax(token_logits, dim=-1)
                        target_token_id = full_inputs['input_ids'][0, i]
                        token_logprob = torch.log(token_probs[target_token_id] + 1e-10)
                        label_logprobs.append(token_logprob.item())
                
                if label_logprobs:
                    total_logprob = sum(label_logprobs)
                else:
                    total_logprob = -100.0
                
                logprobs[label] = float(total_logprob)
                
    except Exception as e:
        print(f"Warning: Could not extract sequence logprobs: {e}")
        import traceback
        traceback.print_exc()
        for label in candidate_labels:
            logprobs[label] = None
    
    return logprobs


def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model response, handling markdown code blocks and other formatting"""
    
    response_text = response_text.strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
    
    lines = response_text.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{') or in_json:
            in_json = True
            json_lines.append(line)
            brace_count += stripped.count('{') - stripped.count('}')
            if brace_count == 0 and in_json:
                break
    
    if json_lines:
        response_text = '\n'.join(json_lines)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
    
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    if start_idx != -1 and end_idx > start_idx:
        json_str = response_text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    print(f"[DEBUG] Failed to parse JSON from response: {response_text[:500]}")
    return None


def create_zero_shot_result(
    text: str,
    candidate_labels: List[str],
    model_response: str,
    logprobs: Optional[Dict[str, float]] = None,
    abstain_threshold: Optional[float] = None,
    label_only_mode: bool = False
) -> ZeroShotResult:
    """
    Create a schema-locked ZeroShotResult from model response.
    
    New Architecture (label_only_mode=True):
    - Model outputs only the label (no JSON)
    - Backend computes confidence scores from logprobs
    - Avoids hallucinated scores from small models
    
    Legacy Architecture (label_only_mode=False):
    - Model outputs JSON with scores
    - Ensures strict JSON schema compliance
    - Raises ValueError if JSON parsing fails
    """
    
    if label_only_mode:
        # NEW ARCHITECTURE: Label-only mode with backend logprob scoring
        # Normalize and validate model output against candidate labels
        import re
        
        # Normalize output: strip punctuation, lowercase
        predicted_label_normalized = re.sub(r'[^\w\s]', '', model_response.strip().lower())
        
        # Find matching candidate label with strict exact matching
        matched_label = None
        for label in candidate_labels:
            label_normalized = re.sub(r'[^\w\s]', '', label.lower())
            if label_normalized == predicted_label_normalized:
                matched_label = label
                break
        
        # If no exact match, trust the model output (don't silently default to first label)
        if matched_label is None:
            print(f"[WARNING] Model output '{model_response}' doesn't exactly match any candidate labels. Trusting model output.")
            # Use the original model response as the predicted label
            matched_label = model_response.strip()
        
        # Compute scores from logprobs (ground truth confidence)
        # Guard against missing logprob entries
        if logprobs:
            import math
            
            # Filter to only labels that have logprobs
            valid_labels = [label for label in candidate_labels if label in logprobs and logprobs[label] is not None]
            
            if valid_labels:
                # Convert logprobs to probabilities using softmax (only for valid labels)
                logprob_values = [logprobs[label] for label in valid_labels]
                max_logprob = max(logprob_values)
                
                # Numerically stable softmax
                exp_values = [math.exp(lp - max_logprob) for lp in logprob_values]
                sum_exp = sum(exp_values)
                probabilities = [exp_val / sum_exp for exp_val in exp_values]
                
                # Build labels with normalized scores
                labels_with_scores = []
                for label, prob, lp in zip(valid_labels, probabilities, logprob_values):
                    labels_with_scores.append(
                        ClassificationLabel(
                            label=label,
                            score=prob,
                            logprob=lp
                        )
                    )
                
                # Sort by score (highest first)
                labels_with_scores.sort(key=lambda x: x.score, reverse=True)
                top_label = labels_with_scores[0].label
                top_score = labels_with_scores[0].score
            else:
                # Fallback: logprobs exist but all are None/invalid
                print(f"[WARNING] Logprobs unavailable for all labels. Using matched label with uniform score.")
                labels_with_scores = []
                
                # Ensure matched label appears in results even if not in candidates
                all_labels = candidate_labels if matched_label in candidate_labels else [matched_label] + candidate_labels
                
                for label in all_labels:
                    score = 1.0 / len(all_labels)
                    labels_with_scores.append(
                        ClassificationLabel(
                            label=label,
                            score=score,
                            logprob=None
                        )
                    )
                top_label = matched_label
                top_score = 1.0 / len(all_labels)
        else:
            # Fallback: no logprobs computed at all
            print(f"[WARNING] Logprobs not computed. Using matched label with uniform scores.")
            labels_with_scores = []
            
            # Ensure matched label appears in results even if not in candidates
            all_labels = candidate_labels if matched_label in candidate_labels else [matched_label] + candidate_labels
            
            for label in all_labels:
                score = 1.0 / len(all_labels)
                labels_with_scores.append(
                    ClassificationLabel(
                        label=label,
                        score=score,
                        logprob=None
                    )
                )
            top_label = matched_label
            top_score = 1.0 / len(all_labels)
    else:
        # LEGACY ARCHITECTURE: JSON parsing mode
        parsed = parse_json_response(model_response)
        
        if parsed is None:
            raise ValueError(
                f"Failed to parse JSON from model response. "
                f"Model returned: {model_response[:200]}... "
                f"Expected JSON with 'labels' array containing label/score pairs."
            )
        
        labels_with_scores = []
        for label_data in parsed.get("labels", []):
            label = label_data.get("label", "")
            score = float(label_data.get("score", 0.0))
            
            logprob_val = None
            if logprobs and label in logprobs:
                logprob_val = logprobs[label]
            
            labels_with_scores.append(
                ClassificationLabel(
                    label=label,
                    score=score,
                    logprob=logprob_val
                )
            )
        
        labels_with_scores.sort(key=lambda x: x.score, reverse=True)
        
        if labels_with_scores:
            top_label = labels_with_scores[0].label
            top_score = labels_with_scores[0].score
        else:
            top_label = parsed.get("top_label", candidate_labels[0] if candidate_labels else "unknown")
            top_score = float(parsed.get("top_score", 0.0))
    
    # Check abstain threshold
    should_abstain = False
    if abstain_threshold is not None and top_score < abstain_threshold:
        should_abstain = True
    
    return ZeroShotResult(
        text=text,
        labels=labels_with_scores,
        top_label=top_label,
        top_score=top_score,
        should_abstain=should_abstain,
        abstain_threshold=abstain_threshold
    )


class LLMZeroShotClassifier:
    """
    Zero-shot classifier using loaded LLM models with:
    1. Schema-locked JSON outputs for structured predictions
    2. Logprob-based scoring for confidence calibration
    """
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def classify(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template: str = "This text is about {label}.",
        use_logprobs: bool = True,
        abstain_threshold: Optional[float] = None,
        max_tokens: int = 300,
        temperature: float = 0.1
    ) -> ZeroShotResult:
        """
        Perform zero-shot classification with logprob-based scoring.
        
        New Architecture:
        - Model outputs only the label (no hallucinated scores)
        - Backend computes real confidence scores from logprobs
        - JSON response assembled server-side with calibrated scores
        
        Args:
            text: Input text to classify
            candidate_labels: List of possible labels
            hypothesis_template: Template for classification
            use_logprobs: Whether to use logprob-based scoring (recommended: True)
            abstain_threshold: Confidence threshold below which to abstain from classification
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            ZeroShotResult with backend-computed confidence scores
        """
        
        # Use simplified label-only prompt (avoids hallucinated scores)
        prompt = build_label_only_prompt(text, candidate_labels)
        
        # Use chat template if available (for instruction-tuned models like SmolLM2)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate just the label (short output, ~1-10 tokens)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # Reduced - we only need label name
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
        
        response_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        print(f"[DEBUG] Model predicted label: {response_text}")
        
        # Conditionally compute logprobs based on user preference
        logprobs = None
        if use_logprobs:
            # Compute real confidence scores from model internal probabilities
            # This is the ground truth - not hallucinated by the model
            logprobs = extract_logprobs_from_sequences(
                self.model,
                self.tokenizer,
                text,
                candidate_labels,
                hypothesis_template,
                self.device
            )
            print(f"[DEBUG] Logprobs computed: {logprobs}")
        else:
            # Skip expensive logprob computation - use uniform scores
            print(f"[DEBUG] Logprobs disabled - using uniform scores for speed")
        
        result = create_zero_shot_result(
            text=text,
            candidate_labels=candidate_labels,
            model_response=response_text,
            logprobs=logprobs,
            abstain_threshold=abstain_threshold,
            label_only_mode=True  # New parameter indicating backend scoring
        )
        
        return result
