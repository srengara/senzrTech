"""
Training Module
===============

Train ResNet34-1D model for glucose prediction from PPG signals.

Classes:
- ResNet34_1D: 34-layer residual network architecture
- GlucosePredictor: High-level interface for glucose prediction
- ResidualBlock1D: Basic building block

Scripts:
- train_glucose_predictor.py: Complete training script with validation
"""

from .resnet34_glucose_predictor import (
    ResidualBlock1D,
    ResNet34_1D,
    GlucosePredictor
)

__all__ = [
    'ResidualBlock1D',
    'ResNet34_1D',
    'GlucosePredictor'
]
