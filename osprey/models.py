from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, channel_size: int) -> None:
        """Very small CNN for testing code"""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel_size, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Assumes targets are already one-hot encoded
        targets = targets.float() if targets.dtype != torch.float32 else targets
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1 - targets) * (1 - prob)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_factor = alpha_factor * (1 - p_t) ** self.gamma
        loss = focal_factor * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Focal cross entropy loss for handling class imbalance.
        
        Applies a modulating term (1 - pt)^gamma to the cross entropy loss to down-weight
        easy examples and focus on hard negatives.
        
        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples
                   or a list of weights for each class. Default: 0.25
            gamma: Exponent of the modulating factor (1 - pt)^gamma. Default: 2.0
            reduction: Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output of shape (B, C) where C is number of classes
            targets: Ground truth labels of shape (B,) with class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p = torch.softmax(logits, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
