"""
Custom metrics for ConvLEM wildfire prediction.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from pyhazards.metrics import MetricBase


class WildfireAccuracy(MetricBase):
    """
    Binary accuracy for wildfire predictions.
    
    Computes accuracy after applying sigmoid threshold (default: 0.5).
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self) -> None:
        """Reset metric state."""
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metric with batch predictions.
        
        Args:
            preds: Model logits (batch, num_counties)
            targets: Ground truth labels (batch, num_counties)
        """
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """Compute final accuracy."""
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        
        # Apply sigmoid and threshold
        probs = torch.sigmoid(preds)
        pred_labels = (probs > self.threshold).float()
        
        # Calculate accuracy
        correct = (pred_labels == targets).float().sum()
        total = targets.numel()
        accuracy = (correct / total).item()
        
        return {"Accuracy": accuracy}


class WildfirePrecisionRecallF1(MetricBase):
    """
    Precision, Recall, and F1 Score for wildfire predictions.
    
    All three metrics computed together for efficiency.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self) -> None:
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        
        # Apply sigmoid and threshold
        probs = torch.sigmoid(preds)
        pred_labels = (probs > self.threshold).float()
        
        # Calculate TP, FP, FN
        tp = ((pred_labels == 1) & (targets == 1)).float().sum().item()
        fp = ((pred_labels == 1) & (targets == 0)).float().sum().item()
        fn = ((pred_labels == 0) & (targets == 1)).float().sum().item()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }


class CountyRiskMetrics(MetricBase):
    """
    County-level risk metrics.
    
    Tracks:
    - Average number of at-risk counties per sample
    - Percentage of at-risk counties correctly identified
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self) -> None:
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        
        # Apply sigmoid and threshold
        probs = torch.sigmoid(preds)
        pred_labels = (probs > self.threshold).float()
        
        # Average at-risk counties per sample
        avg_predicted = pred_labels.sum(dim=1).mean().item()
        avg_actual = targets.sum(dim=1).mean().item()
        
        # Coverage: what fraction of at-risk counties were caught?
        at_risk_mask = targets == 1
        if at_risk_mask.sum() > 0:
            coverage = (pred_labels[at_risk_mask] == 1).float().mean().item()
        else:
            coverage = 0.0
        
        return {
            "AvgPredictedRisk": avg_predicted,
            "AvgActualRisk": avg_actual,
            "RiskCoverage": coverage,
        }


__all__ = [
    "WildfireAccuracy",
    "WildfirePrecisionRecallF1",
    "CountyRiskMetrics",
]