import json
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F


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
    
    prompt = f"""You are a precise text classifier. Classify the following text into one of the given labels.

TEXT TO CLASSIFY:
{text}

CANDIDATE LABELS:
{labels_str}

INSTRUCTIONS:
1. Analyze the text carefully
2. Assign a confidence score (0.0 to 1.0) for each label
3. Scores should sum to approximately 1.0
4. Return your response as valid JSON matching this exact schema:

{{
  "text": "{text[:50]}...",
  "labels": [
    {{"label": "label1", "score": 0.0}},
    {{"label": "label2", "score": 0.0}}
  ],
  "top_label": "most_confident_label",
  "top_score": 0.0
}}

Return ONLY the JSON object, no other text."""
    
    return prompt


def extract_logprobs_from_model(
    model,
    tokenizer,
    text: str,
    candidate_labels: List[str],
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Extract log probabilities for each label using the model's output logits.
    This provides calibrated confidence scores.
    """
    logprobs = {}
    
    try:
        for label in candidate_labels:
            prompt = f"Text: {text}\nThis text is classified as: {label}"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                if len(logits.shape) == 3:
                    last_token_logits = logits[0, -1, :]
                else:
                    last_token_logits = logits[-1, :]
                
                probs = F.softmax(last_token_logits, dim=-1)
                avg_prob = probs.mean().item()
                logprob = float(torch.log(torch.tensor(avg_prob)).item())
                
                logprobs[label] = logprob
                
    except Exception as e:
        print(f"Warning: Could not extract logprobs: {e}")
        for label in candidate_labels:
            logprobs[label] = None
    
    return logprobs


def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model response, handling markdown code blocks and other formatting"""
    
    response_text = response_text.strip()
    
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
    
    lines = response_text.split('\n')
    json_lines = []
    in_json = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{') or in_json:
            in_json = True
            json_lines.append(line)
            if stripped.endswith('}') and json_lines:
                break
    
    if json_lines:
        response_text = '\n'.join(json_lines)
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
    
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
    """
    
    parsed = parse_json_response(model_response)
    
    if parsed is None:
        scores = [1.0 / len(candidate_labels)] * len(candidate_labels)
        parsed = {
            "labels": [
                {"label": label, "score": score}
                for label, score in zip(candidate_labels, scores)
            ],
            "top_label": candidate_labels[0],
            "top_score": scores[0]
        }
    
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
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=use_logprobs
            )
        
        response_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        logprobs = None
        if use_logprobs:
            logprobs = extract_logprobs_from_model(
                self.model,
                self.tokenizer,
                text,
                candidate_labels,
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
