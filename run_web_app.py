#!/usr/bin/env python
"""
Run VitalDB PPG Analysis Web Application

Usage:
    python run_web_app.py

Then open: http://localhost:5000
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run web app
from src.web_app.web_app import app

if __name__ == '__main__':
    print("=" * 70)
    print("VitalDB PPG Analysis Web Interface")
    print("=" * 70)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)
