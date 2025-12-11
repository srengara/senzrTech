"""
Data Extraction Module
======================

Extract and process PPG and glucose data from VitalDB.

Classes:
- PPGExtractor: Extract PPG signals
- GlucoseExtractor: Extract glucose measurements
- PPGSegmenter: Preprocess and segment PPG signals

Functions:
- ppg_peak_detection_pipeline_with_template: Detect peaks with quality filtering
"""

from .ppg_extractor import PPGExtractor
from .glucose_extractor import GlucoseExtractor
from .ppg_segmentation import PPGSegmenter
from .peak_detection import (
    ppg_peak_detection_pipeline,
    ppg_peak_detection_pipeline_with_template,
    compute_template,
    filter_windows_by_similarity,
    cosine_similarity
)

__all__ = [
    'PPGExtractor',
    'GlucoseExtractor',
    'PPGSegmenter',
    'ppg_peak_detection_pipeline',
    'ppg_peak_detection_pipeline_with_template',
    'compute_template',
    'filter_windows_by_similarity',
    'cosine_similarity'
]
