"""
Web Application Module
=======================

Flask-based web interface for PPG analysis and glucose prediction pipeline.

5-Step Pipeline:
1. Select Case & Track - Choose VitalDB case and PPG track
2. View Raw Data - Visualize original signal
3. View Cleansed Data - See filtered signal
4. Peak Detection - Detect and filter heartbeats
5. Glucose Labels - Extract glucose from VitalDB or enter manually

Run:
    python -m src.web_app.web_app
    # Then open http://localhost:5000
"""

__all__ = []
