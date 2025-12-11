"""
VitalDB PPG Glucose Prediction Package
======================================

Main package for PPG signal processing and glucose prediction from VitalDB dataset.

Modules:
- data_extraction: Extract and process PPG and glucose data from VitalDB
- training: Train ResNet34-1D model on PPG-glucose pairs
- inference: Run predictions on new PPG data
- web_app: Web interface for data processing pipeline
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "VitalDB PPG Glucose Team"

# Import key classes for convenience
from src.data_extraction.ppg_extractor import PPGExtractor
from src.data_extraction.glucose_extractor import GlucoseExtractor
from src.training.resnet34_glucose_predictor import ResNet34_1D, GlucosePredictor
from src.inference.glucose_from_csv import predict_glucose_from_csv

__all__ = [
    'PPGExtractor',
    'GlucoseExtractor',
    'ResNet34_1D',
    'GlucosePredictor',
    'predict_glucose_from_csv'
]
