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


def build_zero_shot_prompt(
    text: str,
    candidate_labels: List[str],
    hypothesis_template: str = "This text is about {label}."
) -> str:
    """Build a prompt for zero-shot classification with JSON schema enforcement"""
    
    labels_str = json.dumps(candidate_labels)
    
    hypotheses_examples = []
    for label in candidate_labels[:3]:
        hypothesis = hypothesis_template.replace("{label}", label)
        hypotheses_examples.append(f'  - For label "{label}": "{hypothesis}"')
    hypotheses_str = "\n".join(hypotheses_examples)
    
    prompt = f"""Classify this text into one category. Reply with valid JSON only.

Text: {text}

Categories: {labels_str}

Output format (JSON only):
{{
  "labels": [
    {{"label": "{candidate_labels[0]}", "score": 0.7}},
    {{"label": "{candidate_labels[1] if len(candidate_labels) > 1 else candidate_labels[0]}", "score": 0.2}},
    {{"label": "{candidate_labels[2] if len(candidate_labels) > 2 else candidate_labels[0]}", "score": 0.1}}
  ],
  "top_label": "{candidate_labels[0]}",
  "top_score": 0.7
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
    abstain_threshold: Optional[float] = None
) -> ZeroShotResult:
    """
    Create a schema-locked ZeroShotResult from model response.
    Ensures strict JSON schema compliance.
    Raises ValueError if JSON parsing fails.
    """
    
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
        Perform zero-shot classification with schema-locked JSON output.
        
        Args:
            text: Input text to classify
            candidate_labels: List of possible labels
            hypothesis_template: Template for classification (not used in current implementation)
            use_logprobs: Whether to extract and include log probabilities
            abstain_threshold: Confidence threshold below which to abstain from classification
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            ZeroShotResult with schema-locked JSON structure
        """
        
        prompt = build_zero_shot_prompt(text, candidate_labels, hypothesis_template)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Create early stopping criteria for valid JSON
        stopping_criteria = [JSONCompletionStoppingCriteria(
            tokenizer=self.tokenizer,
            prompt_length=inputs['input_ids'].shape[1]
        )]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=use_logprobs,
                stopping_criteria=stopping_criteria
            )
        
        response_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        logprobs = None
        if use_logprobs:
            logprobs = extract_logprobs_from_sequences(
                self.model,
                self.tokenizer,
                text,
                candidate_labels,
                hypothesis_template,
                self.device
            )
        
        result = create_zero_shot_result(
            text=text,
            candidate_labels=candidate_labels,
            model_response=response_text,
            logprobs=logprobs,
            abstain_threshold=abstain_threshold
        )
        
        return result
