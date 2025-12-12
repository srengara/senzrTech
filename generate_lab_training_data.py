#!/usr/bin/env python
"""
Lab PPG Data Processor for Inference
=====================================
Standalone application to generate ppg_windows.csv from lab PPG data (time, ppg format)
for inference.

This script processes raw PPG data through the complete pipeline:
1. Data loading and cleansing
2. Signal preprocessing (filtering, smoothing)
3. Peak detection
4. Window extraction
5. Template-based window filtering (deduplication)
6. Output generation for inference

Usage:
    # Basic usage with CSV input
    python generate_lab_training_data.py --input lab_ppg.csv --sampling_rate 100

    # Specify output directory
    python generate_lab_training_data.py --input lab_ppg.csv --sampling_rate 100 --output ./inference_data

    # Adjust peak detection parameters
    python generate_lab_training_data.py --input lab_ppg.csv --sampling_rate 100 \
        --height 0.4 --distance 0.9 --similarity 0.9

Input Format:
    CSV file with columns: time, ppg
    - time: timestamp in seconds
    - ppg: PPG amplitude values
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, resample

# Add src to path if available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data_extraction.peak_detection import ppg_peak_detection_pipeline_with_template
except ImportError:
    # Inline implementation if module not found
    from typing import List, Tuple, Optional

    def detect_peaks(ppg_signal: np.ndarray, height_threshold: float = 20,
                    distance_threshold: Optional[float] = None, fs: float = 100) -> List[int]:
        """Detect peaks in PPG signal."""
        if distance_threshold is None:
            distance_threshold = 0.8 * fs

        peaks = []
        for i in range(1, len(ppg_signal) - 1):
            if ppg_signal[i - 1] < ppg_signal[i] > ppg_signal[i + 1]:
                if ppg_signal[i] > height_threshold:
                    if len(peaks) == 0 or (i - peaks[-1]) > distance_threshold:
                        peaks.append(i)
        return peaks

    def count_peaks(window: np.ndarray, height_threshold: Optional[float] = None) -> int:
        """Count peaks in a window."""
        if len(window) < 3:
            return 0
        if height_threshold is None:
            height_threshold = np.median(window)
        count = 0
        for i in range(1, len(window) - 1):
            if window[i - 1] < window[i] > window[i + 1] and window[i] > height_threshold:
                count += 1
        return count

    def extract_windows(ppg_signal: np.ndarray, peaks: List[int], window_size: int,
                       similarity_threshold: float = 0.85) -> List[np.ndarray]:
        """Extract windows around peaks."""
        windows = []
        for peak in peaks:
            window_start = max(0, peak - window_size // 2)
            window_end = min(len(ppg_signal), peak + window_size // 2)
            window = ppg_signal[window_start:window_end]
            if count_peaks(window) == 1:
                windows.append(window)
        return windows

    def compute_template(windows: List[np.ndarray]) -> np.ndarray:
        """Compute average template from windows."""
        if not windows:
            return np.array([])
        lengths = [len(w) for w in windows]
        most_common_length = max(set(lengths), key=lengths.count)
        filtered_windows = [w for w in windows if len(w) == most_common_length]
        if not filtered_windows:
            return np.array([])
        stacked_windows = np.stack(filtered_windows, axis=0)
        template = np.mean(stacked_windows, axis=0)
        return template

    def cosine_similarity(window: np.ndarray, template: np.ndarray) -> float:
        """Compute cosine similarity between window and template."""
        if len(window) != len(template):
            min_len = min(len(window), len(template))
            window = window[:min_len]
            template = template[:min_len]
        dot_product = np.sum(window * template)
        magnitude_window = np.sqrt(np.sum(window ** 2))
        magnitude_template = np.sqrt(np.sum(template ** 2))
        if magnitude_window == 0 or magnitude_template == 0:
            return 0.0
        similarity = dot_product / (magnitude_window * magnitude_template)
        return similarity

    def filter_windows_by_similarity(windows: List[np.ndarray], template: np.ndarray,
                                    similarity_threshold: float = 0.85) -> List[np.ndarray]:
        """Filter windows by similarity to template."""
        filtered_windows = []
        for window in windows:
            similarity = cosine_similarity(window, template)
            if similarity >= similarity_threshold:
                filtered_windows.append(window)
        return filtered_windows

    def ppg_peak_detection_pipeline_with_template(
        ppg_signal: np.ndarray, fs: float = 100, window_duration: float = 1,
        height_threshold: float = 20, distance_threshold: Optional[float] = None,
        similarity_threshold: float = 0.85
    ) -> Tuple[List[int], List[np.ndarray], np.ndarray, List[np.ndarray]]:
        """Complete PPG peak detection pipeline with template filtering."""
        if distance_threshold is None:
            distance_threshold = 0.8 * fs
        window_size = int(fs * window_duration)

        peaks = detect_peaks(ppg_signal, height_threshold=height_threshold,
                           distance_threshold=distance_threshold, fs=fs)
        windows = extract_windows(ppg_signal, peaks, window_size=window_size,
                                similarity_threshold=similarity_threshold)
        template = compute_template(windows)
        filtered_windows = filter_windows_by_similarity(windows, template,
                                                       similarity_threshold=similarity_threshold)

        return peaks, filtered_windows, template, windows

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_signal(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Preprocess PPG signal.

    Steps:
    1. Remove DC component
    2. Bandpass filter (0.5 - 10 Hz)
    3. Smooth signal with Savitzky-Golay filter

    Args:
        signal: Raw PPG signal
        sampling_rate: Sampling rate in Hz

    Returns:
        Preprocessed signal
    """
    # Remove DC component
    signal_centered = signal - np.mean(signal)

    # Bandpass filter: 0.5 - 10 Hz
    nyquist = sampling_rate / 2
    low_cutoff = 0.5 / nyquist
    high_cutoff = 10.0 / nyquist

    if high_cutoff >= 1.0:
        high_cutoff = 0.99

    try:
        b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
        signal_filtered = filtfilt(b, a, signal_centered)
    except Exception as e:
        logger.warning(f"Filtering failed ({e}), using centered signal")
        signal_filtered = signal_centered

    # Smooth with Savitzky-Golay filter
    window_length = int(sampling_rate * 0.05)  # 50ms window
    if window_length % 2 == 0:
        window_length += 1
    if window_length < 5:
        window_length = 5

    try:
        signal_smoothed = savgol_filter(signal_filtered, window_length, 3)
    except Exception:
        signal_smoothed = signal_filtered

    return signal_smoothed


def generate_lab_training_data(input_csv, sampling_rate, output_dir,
                               target_sampling_rate=None,
                               height_multiplier=0.3, distance_multiplier=0.8,
                               similarity_threshold=0.85):
    """
    Generate PPG windows file from lab PPG data for inference.

    Args:
        input_csv: Path to input CSV with 'time' and 'ppg' columns
        sampling_rate: Sampling rate in Hz of input data
        output_dir: Output directory for CSV files
        target_sampling_rate: Target sampling rate after downsampling (optional, default=100Hz)
        height_multiplier: Peak detection height threshold multiplier
        distance_multiplier: Peak detection distance threshold multiplier
        similarity_threshold: Template similarity threshold for filtering windows

    Returns:
        Tuple of (ppg_file_path, stats_dict)
    """
    logger.info("=" * 70)
    logger.info("Lab PPG Data Processor for Inference")
    logger.info("=" * 70)
    logger.info(f"Input: {input_csv}")
    logger.info(f"Sampling Rate: {sampling_rate} Hz")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load PPG data
    logger.info("Step 1: Loading PPG data...")
    try:
        df = pd.read_csv(input_csv)

        if 'time' not in df.columns or 'ppg' not in df.columns:
            raise ValueError("CSV must contain 'time' and 'ppg' columns")

        logger.info(f"✓ Loaded {len(df)} samples")
    except Exception as e:
        logger.error(f"✗ Failed to load PPG data: {e}")
        raise

    # Step 2: Cleanse data
    logger.info("\nStep 2: Cleansing PPG data...")
    df = df.dropna(subset=['ppg', 'time'])

    time = df['time'].values
    signal = df['ppg'].values

    logger.info(f"✓ Cleansed data: {len(signal)} samples")
    logger.info(f"  Duration: {time[-1] - time[0]:.2f} seconds")

    # Verify sampling rate
    if len(time) > 1:
        time_diffs = np.diff(time)
        median_interval = np.median(time_diffs)
        estimated_sr = 1.0 / median_interval if median_interval > 0 else sampling_rate

        if abs(estimated_sr - sampling_rate) > sampling_rate * 0.1:
            logger.warning(f"  Detected sampling rate: {estimated_sr:.2f} Hz (using {sampling_rate} Hz)")

    # Step 2.5: Downsample if needed
    if target_sampling_rate is None:
        target_sampling_rate = 100  # Default to 100 Hz as per paper

    if sampling_rate > target_sampling_rate:
        logger.info(f"\nStep 2.5: Downsampling from {sampling_rate} Hz to {target_sampling_rate} Hz...")

        # Calculate downsampling factor
        downsample_factor = int(sampling_rate / target_sampling_rate)

        # Downsample signal (anti-aliasing filter applied automatically by resample)
        num_samples_downsampled = int(len(signal) * target_sampling_rate / sampling_rate)
        signal = resample(signal, num_samples_downsampled)
        time = np.linspace(time[0], time[-1], num_samples_downsampled)

        logger.info(f"✓ Downsampled: {len(signal)} samples at {target_sampling_rate} Hz")
        logger.info(f"  Downsample factor: {downsample_factor}x")

        # Update sampling rate for subsequent processing
        sampling_rate = target_sampling_rate
    elif target_sampling_rate and sampling_rate != target_sampling_rate:
        logger.warning(f"  Input sampling rate ({sampling_rate} Hz) is lower than target ({target_sampling_rate} Hz)")
        logger.warning(f"  Using input sampling rate: {sampling_rate} Hz")

    # Step 3: Preprocess signal
    logger.info(f"\nStep 3: Preprocessing signal (at {sampling_rate} Hz)...")
    preprocessed_signal = preprocess_signal(signal, sampling_rate)
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
    if len(all_windows) > 0:
        logger.info(f"  Filtering rate: {len(filtered_windows)/len(all_windows)*100:.1f}%")

    if len(filtered_windows) == 0:
        logger.error("✗ No valid windows extracted. Try adjusting peak detection parameters.")
        raise ValueError("No valid windows extracted")

    num_windows = len(filtered_windows)

    # Step 5: Save PPG windows to CSV
    logger.info("\nStep 5: Saving PPG windows...")
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

    # Step 6: Generate statistics
    logger.info("\nStep 6: Summary Statistics")
    logger.info("=" * 70)

    stats = {
        'input_file': input_csv,
        'num_windows': num_windows,
        'window_length': len(filtered_windows[0]),
        'sampling_rate': sampling_rate,
        'total_peaks': len(peaks),
        'filtered_windows': num_windows,
        'filtering_rate': float(num_windows / len(all_windows) * 100) if len(all_windows) > 0 else 0,
        'ppg_file': ppg_windows_file,
        'signal_duration': float(time[-1] - time[0]),
        'total_samples': len(signal)
    }

    logger.info(f"  Input file: {stats['input_file']}")
    logger.info(f"  Number of windows: {stats['num_windows']}")
    logger.info(f"  Window length: {stats['window_length']} samples")
    logger.info(f"  Sampling rate: {stats['sampling_rate']} Hz")
    logger.info(f"  Total peaks detected: {stats['total_peaks']}")
    logger.info(f"  Windows after filtering: {stats['filtered_windows']}")
    logger.info(f"  Filtering rate: {stats['filtering_rate']:.1f}%")
    logger.info(f"  Signal duration: {stats['signal_duration']:.2f} seconds")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info("")
    logger.info(f"✓ PPG windows file: {stats['ppg_file']}")
    logger.info("=" * 70)
    logger.info("PPG data processing complete!")
    logger.info("=" * 70)

    return ppg_windows_file, stats


def main():
    parser = argparse.ArgumentParser(
        description='Process lab PPG data for inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_lab_training_data.py --input lab_ppg.csv --sampling_rate 100

  # Specify output directory
  python generate_lab_training_data.py --input lab_ppg.csv --sampling_rate 100 --output ./my_data

  # Adjust peak detection parameters
  python generate_lab_training_data.py --input lab_ppg.csv --sampling_rate 100 \\
      --height 0.4 --distance 0.9 --similarity 0.9

Input CSV Format:
  The input CSV must contain two columns:
    - time: timestamp in seconds
    - ppg: PPG amplitude values
        """
    )

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with "time" and "ppg" columns')
    parser.add_argument('--sampling_rate', type=float, required=True,
                        help='Sampling rate in Hz (e.g., 100, 500)')

    # Optional arguments
    parser.add_argument('--output', type=str, default='./inference_data',
                        help='Output directory for CSV files (default: ./inference_data)')
    parser.add_argument('--target_sampling_rate', type=float, default=100,
                        help='Target sampling rate for downsampling in Hz (default: 100)')

    # Peak detection parameters
    parser.add_argument('--height', type=float, default=0.3,
                        help='Peak height threshold multiplier (default: 0.3)')
    parser.add_argument('--distance', type=float, default=0.8,
                        help='Peak distance threshold multiplier (default: 0.8)')
    parser.add_argument('--similarity', type=float, default=0.85,
                        help='Template similarity threshold (default: 0.85)')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        logger.error(f"✗ Input file not found: {args.input}")
        return 1

    try:
        # Process PPG data
        ppg_file, stats = generate_lab_training_data(
            input_csv=args.input,
            sampling_rate=args.sampling_rate,
            output_dir=args.output,
            target_sampling_rate=args.target_sampling_rate,
            height_multiplier=args.height,
            distance_multiplier=args.distance,
            similarity_threshold=args.similarity
        )

        logger.info("\n✅ SUCCESS! PPG windows file is ready for inference.")
        logger.info(f"\nGenerated file: {ppg_file}")
        logger.info(f"Total windows: {stats['num_windows']}")

        return 0

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
