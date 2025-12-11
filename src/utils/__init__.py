"""
Utilities Module
================

Utility functions and helper classes.

Classes:
- VitalDBUtility: Interface to VitalDB API

Scripts:
- ppg_analysis_pipeline.py: Batch processing pipeline
- ppg_peak_detection_pipeline.py: Peak detection utilities
"""

from .vitaldb_utility import VitalDBUtility

__all__ = ['VitalDBUtility']
