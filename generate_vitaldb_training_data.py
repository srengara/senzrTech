#!/usr/bin/env python
"""
VitalDB Training Data Generator
================================
Standalone application to generate training files (ppg_windows.csv and glucose_labels.csv)
from VitalDB case data.

Usage:
    python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0
    python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose auto
    python generate_training_data.py --case_id 1 --track SNUADC/PLETH --output ./training_data

Features:
    - Extracts PPG data from VitalDB
    - Detects peaks and filters windows
    - Generates glucose labels (manual value or auto from clinical data)
    - Outputs training-ready CSV files
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_extraction.ppg_extractor import PPGExtractor
from src.data_extraction.ppg_segmentation import PPGSegmenter
from src.data_extraction.glucose_extractor import GlucoseExtractor
from src.data_extraction.peak_detection import ppg_peak_detection_pipeline_with_template

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_data(case_id, track_name, glucose_value, output_dir,
                          height_multiplier=0.3, distance_multiplier=0.8,
                          similarity_threshold=0.85):
    """
    Generate training data files from VitalDB case.

    Args:
        case_id: VitalDB case ID
        track_name: PPG track name (e.g., 'SNUADC/PLETH')
        glucose_value: Glucose value in mg/dL, or 'auto' to extract from clinical data
        output_dir: Output directory for CSV files
        height_multiplier: Peak detection height threshold multiplier
        distance_multiplier: Peak detection distance threshold multiplier
        similarity_threshold: Template similarity threshold for filtering windows

    Returns:
        Tuple of (ppg_file_path, glucose_file_path, stats_dict)
    """
    logger.info("=" * 70)
    logger.info("VitalDB Training Data Generator")
    logger.info("=" * 70)
    logger.info(f"Case ID: {case_id}")
    logger.info(f"Track: {track_name}")
    logger.info(f"Glucose: {glucose_value}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract PPG data
    logger.info("Step 1: Extracting PPG data from VitalDB...")
    ppg_extractor = PPGExtractor()

    try:
        ppg_result = ppg_extractor.extract_ppg_raw(case_id, track_name, output_dir)
        logger.info(f"✓ Extracted PPG data: {ppg_result['num_samples']} samples")
    except Exception as e:
        logger.error(f"✗ Failed to extract PPG data: {e}")
        raise

    # Step 2: Load and cleanse PPG data
    logger.info("\nStep 2: Loading and cleansing PPG data...")
    df = pd.read_csv(ppg_result['csv_file'])

    # Cleanse data
    df = df.dropna(subset=['ppg'])
    if df['time'].isna().all():
        sampling_rate = ppg_result['expected_sampling_rate']
        if sampling_rate is None:
            raise ValueError("Cannot determine sampling rate for track with all NaN time values")
        df['time'] = np.arange(len(df)) / sampling_rate
    else:
        df = df.dropna(subset=['time'])

    time = df['time'].values
    signal = df['ppg'].values

    logger.info(f"✓ Cleansed data: {len(signal)} samples")

    # Step 3: Preprocess signal
    logger.info("\nStep 3: Preprocessing signal...")
    sampling_rate = ppg_result['expected_sampling_rate']
    if sampling_rate is None:
        raise ValueError("Cannot preprocess signal without a known sampling rate")
    segmenter = PPGSegmenter(sampling_rate=sampling_rate)
    preprocessed_signal = segmenter.preprocess_signal(signal)
    logger.info(f"✓ Signal preprocessed")

    # Step 4: Detect peaks and filter windows
    logger.info("\nStep 4: Detecting peaks and filtering windows...")

    signal_mean = np.mean(preprocessed_signal)
    signal_std = np.std(preprocessed_signal)
    height_threshold = signal_mean + height_multiplier * signal_std
    distance_threshold = distance_multiplier * sampling_rate

    peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
        ppg_signal=preprocessed_signal,
        fs=float(sampling_rate),
        window_duration=1.0,
        height_threshold=float(height_threshold),
        distance_threshold=distance_threshold,
        similarity_threshold=similarity_threshold
    )

    logger.info(f"✓ Detected {len(peaks)} peaks")
    logger.info(f"✓ Extracted {len(all_windows)} windows")
    logger.info(f"✓ Filtered to {len(filtered_windows)} high-quality windows")
    logger.info(f"  Filtering rate: {len(filtered_windows)/len(all_windows)*100:.1f}%")

    if len(filtered_windows) == 0:
        logger.error("✗ No valid windows extracted. Try adjusting peak detection parameters.")
        raise ValueError("No valid windows extracted")

    # Step 5: Get glucose values
    logger.info("\nStep 5: Generating glucose labels...")
    glucose_extractor = GlucoseExtractor()

    if glucose_value == 'auto':
        # Try to get from clinical data
        logger.info("  Attempting to extract glucose from clinical data...")
        clinical_glucose = glucose_extractor.get_clinical_glucose(case_id)

        if clinical_glucose is None:
            logger.error("✗ No preop_glucose found in clinical data")
            raise ValueError("No clinical glucose available. Please provide a manual value.")

        glucose_value = clinical_glucose
        glucose_source = 'clinical:preop_glucose'
        logger.info(f"✓ Using clinical glucose: {glucose_value} mg/dL")
    else:
        # Manual glucose value
        try:
            glucose_value = float(glucose_value)
            glucose_source = 'manual'
            logger.info(f"✓ Using manual glucose: {glucose_value} mg/dL")
        except (ValueError, TypeError):
            logger.error(f"✗ Invalid glucose value: {glucose_value}")
            raise ValueError("Glucose value must be a number or 'auto'")

    # Create glucose array (same value for all windows)
    num_windows = len(filtered_windows)
    glucose_labels = np.full(num_windows, glucose_value)

    logger.info(f"✓ Generated {num_windows} glucose labels")

    # Step 6: Save PPG windows to CSV
    logger.info("\nStep 6: Saving PPG windows...")
    ppg_windows_file = os.path.join(output_dir, 'ppg_windows.csv')

    ppg_rows = []
    for window_idx, window in enumerate(filtered_windows):
        for sample_idx, amplitude in enumerate(window):
            ppg_rows.append({
                'window_index': window_idx,
                'sample_index': sample_idx,
                'amplitude': float(amplitude)
            })

    ppg_df = pd.DataFrame(ppg_rows)
    ppg_df.to_csv(ppg_windows_file, index=False)

    logger.info(f"✓ Saved PPG windows: {ppg_windows_file}")
    logger.info(f"  Format: {len(filtered_windows)} windows × {len(filtered_windows[0])} samples")

    # Step 7: Save glucose labels to CSV
    logger.info("\nStep 7: Saving glucose labels...")
    glucose_file = os.path.join(output_dir, 'glucose_labels.csv')

    glucose_df = pd.DataFrame({
        'window_index': range(num_windows),
        'glucose_mg_dl': glucose_labels
    })
    glucose_df.to_csv(glucose_file, index=False)

    logger.info(f"✓ Saved glucose labels: {glucose_file}")

    # Step 8: Generate statistics
    logger.info("\nStep 8: Summary Statistics")
    logger.info("=" * 70)

    stats = {
        'case_id': case_id,
        'track': track_name,
        'glucose_source': glucose_source,
        'glucose_value': float(glucose_value),
        'num_windows': num_windows,
        'window_length': len(filtered_windows[0]),
        'sampling_rate': sampling_rate,
        'total_peaks': len(peaks),
        'filtered_windows': num_windows,
        'filtering_rate': float(num_windows / len(all_windows) * 100),
        'ppg_file': ppg_windows_file,
        'glucose_file': glucose_file
    }

    logger.info(f"  Case ID: {stats['case_id']}")
    logger.info(f"  Track: {stats['track']}")
    logger.info(f"  Glucose source: {stats['glucose_source']}")
    logger.info(f"  Glucose value: {stats['glucose_value']} mg/dL")
    logger.info(f"  Number of windows: {stats['num_windows']}")
    logger.info(f"  Window length: {stats['window_length']} samples")
    logger.info(f"  Sampling rate: {stats['sampling_rate']} Hz")
    logger.info(f"  Total peaks detected: {stats['total_peaks']}")
    logger.info(f"  Windows after filtering: {stats['filtered_windows']}")
    logger.info(f"  Filtering rate: {stats['filtering_rate']:.1f}%")
    logger.info("")
    logger.info(f"✓ PPG windows file: {stats['ppg_file']}")
    logger.info(f"✓ Glucose labels file: {stats['glucose_file']}")
    logger.info("=" * 70)
    logger.info("Training data generation complete!")
    logger.info("=" * 70)

    return ppg_windows_file, glucose_file, stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data from VitalDB case',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use manual glucose value
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0

  # Auto-extract glucose from clinical data
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose auto

  # Specify output directory
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0 --output ./my_data

  # Adjust peak detection parameters
  python generate_training_data.py --case_id 1 --track SNUADC/PLETH --glucose 95.0 \\
      --height 0.4 --distance 0.9 --similarity 0.9
        """
    )

    # Required arguments
    parser.add_argument('--case_id', type=int, required=True,
                        help='VitalDB case ID (e.g., 1, 2, 3, ...)')
    parser.add_argument('--track', type=str, required=True,
                        help='PPG track name (e.g., SNUADC/PLETH, Primus/PLETH)')
    parser.add_argument('--glucose', type=str, required=True,
                        help='Glucose value in mg/dL (e.g., 95.0) or "auto" to extract from clinical data')

    # Optional arguments
    parser.add_argument('--output', type=str, default='./training_data',
                        help='Output directory for CSV files (default: ./training_data)')

    # Peak detection parameters
    parser.add_argument('--height', type=float, default=0.3,
                        help='Peak height threshold multiplier (default: 0.3)')
    parser.add_argument('--distance', type=float, default=0.8,
                        help='Peak distance threshold multiplier (default: 0.8)')
    parser.add_argument('--similarity', type=float, default=0.85,
                        help='Template similarity threshold (default: 0.85)')

    args = parser.parse_args()

    try:
        # Generate training data
        ppg_file, glucose_file, stats = generate_training_data(
            case_id=args.case_id,
            track_name=args.track,
            glucose_value=args.glucose,
            output_dir=args.output,
            height_multiplier=args.height,
            distance_multiplier=args.distance,
            similarity_threshold=args.similarity
        )

        logger.info("\n✅ SUCCESS! Training data files are ready.")
        logger.info(f"\nTo train the model, run:")
        logger.info(f"  python -m src.training.train_glucose_predictor --data_dir {args.output}")

        return 0

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
