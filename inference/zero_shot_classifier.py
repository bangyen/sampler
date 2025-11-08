import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from transformers.pipelines import pipeline
import torch
import torch.nn.functional as F


class ClassificationLabel(BaseModel):
    label: str = Field(..., description="The classification label")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    logprob: Optional[float] = Field(None, description="Log probability for calibration")
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score must be between 0 and 1')
        return v


class ZeroShotClassificationResult(BaseModel):
    sequence: str = Field(..., description="The input text that was classified")
    labels: List[ClassificationLabel] = Field(..., description="Classification results with scores")
    top_label: str = Field(..., description="The highest scoring label")
    top_score: float = Field(..., ge=0.0, le=1.0, description="The highest score")
    should_abstain: bool = Field(False, description="Whether to abstain based on confidence threshold")
    abstain_threshold: Optional[float] = Field(None, description="Threshold used for abstain decision")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequence": "I love this product!",
                "labels": [
                    {"label": "positive", "score": 0.95, "logprob": -0.05},
                    {"label": "negative", "score": 0.03, "logprob": -3.51},
                    {"label": "neutral", "score": 0.02, "logprob": -3.91}
                ],
                "top_label": "positive",
                "top_score": 0.95,
                "should_abstain": False,
                "abstain_threshold": 0.5
            }
        }


class ZeroShotClassifier:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        use_logprobs: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.use_logprobs = use_logprobs
        self.classifier = None
        self.model = None
        self.tokenizer = None
        
    def load(self):
        if self.classifier is None:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1 if self.device == "cpu" else 0
            )
            
            if self.use_logprobs:
                self.model = self.classifier.model
                self.tokenizer = self.classifier.tokenizer
                
    def _compute_logprobs(
        self,
        sequence: str,
        labels: List[str],
        hypothesis_template: str = "This example is {}."
    ) -> Dict[str, float]:
        if not self.use_logprobs or self.model is None:
            return {}
            
        logprobs = {}
        
        try:
            for label in labels:
                premise = sequence
                hypothesis = hypothesis_template.format(label)
                
                inputs = self.tokenizer(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    probs = F.softmax(logits, dim=-1)
                    
                    entailment_idx = self.model.config.label2id.get("entailment", 2)
                    entailment_prob = probs[0][entailment_idx].item()
                    
                    logprob = torch.log(probs[0][entailment_idx]).item()
                    logprobs[label] = logprob
                    
        except Exception as e:
            print(f"Warning: Could not compute logprobs: {e}")
            return {}
            
        return logprobs
        
    def classify(
        self,
        sequence: str,
        candidate_labels: List[str],
        hypothesis_template: str = "This example is {}.",
        multi_label: bool = False,
        abstain_threshold: Optional[float] = None
    ) -> ZeroShotClassificationResult:
        if self.classifier is None:
            self.load()
            
        result = self.classifier(
            sequence,
            candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label
        )
        
        logprobs = {}
        if self.use_logprobs:
            logprobs = self._compute_logprobs(
                sequence,
                candidate_labels,
                hypothesis_template
            )
        
        labels_with_scores = []
        for label, score in zip(result['labels'], result['scores']):
            label_obj = ClassificationLabel(
                label=label,
                score=float(score),
                logprob=logprobs.get(label)
            )
            labels_with_scores.append(label_obj)
        
        top_label = result['labels'][0]
        top_score = float(result['scores'][0])
        
        should_abstain = False
        if abstain_threshold is not None and top_score < abstain_threshold:
            should_abstain = True
        
        return ZeroShotClassificationResult(
            sequence=sequence,
            labels=labels_with_scores,
            top_label=top_label,
            top_score=top_score,
            should_abstain=should_abstain,
            abstain_threshold=abstain_threshold
        )
    
    def classify_batch(
        self,
        sequences: List[str],
        candidate_labels: List[str],
        hypothesis_template: str = "This example is {}.",
        multi_label: bool = False,
        abstain_threshold: Optional[float] = None
    ) -> List[ZeroShotClassificationResult]:
        results = []
        for sequence in sequences:
            result = self.classify(
                sequence,
                candidate_labels,
                hypothesis_template,
                multi_label,
                abstain_threshold
            )
            results.append(result)
        return results


ZERO_SHOT_MODELS = {
    "BART Large MNLI": {
        "id": "facebook/bart-large-mnli",
        "params": "406M",
        "memory": "~1.6GB",
        "description": "High-quality zero-shot classification with BART (recommended)",
        "supports_logprobs": True,
    },
    "DeBERTa v3 Large MNLI": {
        "id": "microsoft/deberta-v3-large",
        "params": "434M",
        "memory": "~1.7GB",
        "description": "State-of-the-art accuracy with DeBERTa v3",
        "supports_logprobs": True,
    },
    "DeBERTa v3 Base MNLI": {
        "id": "microsoft/deberta-v3-base",
        "params": "184M",
        "memory": "~700MB",
        "description": "Fast and accurate with DeBERTa v3 base",
        "supports_logprobs": True,
    },
}


def create_classifier(model_name: str = "BART Large MNLI", use_logprobs: bool = True) -> ZeroShotClassifier:
    if model_name not in ZERO_SHOT_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(ZERO_SHOT_MODELS.keys())}")
    
    model_id = ZERO_SHOT_MODELS[model_name]["id"]
    classifier = ZeroShotClassifier(
        model_name=model_id,
        use_logprobs=use_logprobs
    )
    
    return classifier
