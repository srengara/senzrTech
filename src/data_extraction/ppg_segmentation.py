"""
PPG Precision Interval Segmentation
====================================
Implements pulse segmentation algorithm for PPG signals.

Based on precision interval detection using:
- Peak detection
- Valley detection
- Quality assessment
- Interval validation
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict, Optional
import json


class PPGSegmenter:
    """
    PPG signal segmentation using precision interval detection.

    Implements pulse-by-pulse segmentation with quality assessment.
    """

    def __init__(self,
                 sampling_rate: float = 500.0,
                 min_heart_rate: float = 40.0,
                 max_heart_rate: float = 180.0):
        """
        Initialize segmenter.

        Args:
            sampling_rate: Sampling rate in Hz
            min_heart_rate: Minimum expected heart rate (bpm)
            max_heart_rate: Maximum expected heart rate (bpm)
        """
        self.sampling_rate = sampling_rate
        self.min_heart_rate = min_heart_rate
        self.max_heart_rate = max_heart_rate

        # Calculate expected interval ranges
        self.min_interval = 60.0 / max_heart_rate  # seconds
        self.max_interval = 60.0 / min_heart_rate  # seconds

        # Quality thresholds
        self.quality_threshold = 0.5
        self.amplitude_threshold_percentile = 20

    def load_ppg_data(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load PPG data from CSV file.

        Args:
            csv_file: Path to CSV file

        Returns:
            Tuple of (time array, signal array)
        """
        df = pd.read_csv(csv_file)

        if 'time' not in df.columns or 'ppg' not in df.columns:
            raise ValueError("CSV must contain 'time' and 'ppg' columns")

        time = df['time'].values
        signal = df['ppg'].values

        return time, signal

    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess PPG signal.

        Steps:
        1. Remove DC component
        2. Bandpass filter (0.5 - 10 Hz)
        3. Smooth signal

        Args:
            signal: Raw PPG signal

        Returns:
            Preprocessed signal
        """
        # Remove DC component
        signal_centered = signal - np.mean(signal)

        # Bandpass filter: 0.5 - 10 Hz
        # This removes baseline wander and high-frequency noise
        nyquist = self.sampling_rate / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 10.0 / nyquist

        if high_cutoff >= 1.0:
            high_cutoff = 0.99

        try:
            b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
            signal_filtered = filtfilt(b, a, signal_centered)
        except Exception as e:
            print(f"Warning: Filtering failed ({e}), using centered signal")
            signal_filtered = signal_centered

        # Smooth with Savitzky-Golay filter
        window_length = int(self.sampling_rate * 0.05)  # 50ms window
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 5:
            window_length = 5

        try:
            signal_smoothed = savgol_filter(signal_filtered, window_length, 3)
        except Exception:
            signal_smoothed = signal_filtered

        return signal_smoothed

    def detect_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect systolic peaks in PPG signal.

        Args:
            signal: Preprocessed PPG signal

        Returns:
            Array of peak indices
        """
        # Calculate minimum distance between peaks
        min_distance = int(self.min_interval * self.sampling_rate)

        # Calculate height threshold
        signal_std = np.std(signal)
        height_threshold = np.mean(signal) + 0.3 * signal_std

        # Find peaks
        peaks, properties = find_peaks(
            signal,
            height=height_threshold,
            distance=min_distance,
            prominence=signal_std * 0.2
        )

        return peaks

    def detect_valleys(self, signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """
        Detect diastolic valleys (onset points) in PPG signal.

        Args:
            signal: Preprocessed PPG signal
            peaks: Array of peak indices

        Returns:
            Array of valley indices
        """
        valleys = []

        # Invert signal to find valleys as peaks
        inverted_signal = -signal

        for i in range(len(peaks) - 1):
            # Search for valley between consecutive peaks
            start_idx = peaks[i]
            end_idx = peaks[i + 1]

            if end_idx - start_idx < 10:  # Too close
                continue

            # Find minimum in this region
            search_region = inverted_signal[start_idx:end_idx]
            local_valley_idx = np.argmax(search_region)
            valley_idx = start_idx + local_valley_idx

            valleys.append(valley_idx)

        return np.array(valleys)

    def calculate_pulse_quality(self, signal: np.ndarray,
                                onset_idx: int,
                                peak_idx: int,
                                next_onset_idx: int) -> float:
        """
        Calculate quality metric for a single pulse.

        Quality criteria:
        1. Amplitude (peak - onset)
        2. Shape regularity
        3. Duration validity

        Args:
            signal: PPG signal
            onset_idx: Onset (valley) index
            peak_idx: Peak index
            next_onset_idx: Next onset index

        Returns:
            Quality score (0-1, higher is better)
        """
        pulse = signal[onset_idx:next_onset_idx]

        if len(pulse) < 10:
            return 0.0

        # 1. Amplitude quality
        amplitude = signal[peak_idx] - signal[onset_idx]
        amplitude_percentile = np.percentile(signal, self.amplitude_threshold_percentile)
        amplitude_score = min(amplitude / (amplitude_percentile + 1e-6), 1.0)

        # 2. Shape quality - check for monotonic rise and fall
        rise_phase = signal[onset_idx:peak_idx]
        fall_phase = signal[peak_idx:next_onset_idx]

        if len(rise_phase) > 0 and len(fall_phase) > 0:
            # Check if rise is mostly increasing
            rise_diffs = np.diff(rise_phase)
            rise_score = np.sum(rise_diffs > 0) / len(rise_diffs) if len(rise_diffs) > 0 else 0

            # Check if fall is mostly decreasing
            fall_diffs = np.diff(fall_phase)
            fall_score = np.sum(fall_diffs < 0) / len(fall_diffs) if len(fall_diffs) > 0 else 0

            shape_score = (rise_score + fall_score) / 2
        else:
            shape_score = 0.0

        # 3. Duration validity
        duration = (next_onset_idx - onset_idx) / self.sampling_rate
        if self.min_interval <= duration <= self.max_interval:
            duration_score = 1.0
        else:
            duration_score = 0.3  # Penalize but don't completely reject

        # Combine scores
        quality = (amplitude_score * 0.4 + shape_score * 0.4 + duration_score * 0.2)

        return quality

    def segment_pulses(self, time: np.ndarray,
                      signal: np.ndarray) -> Dict:
        """
        Segment PPG signal into individual pulses.

        Algorithm:
        1. Preprocess signal
        2. Detect peaks (systolic points)
        3. Detect valleys (diastolic/onset points)
        4. Calculate quality for each pulse
        5. Extract intervals

        Args:
            time: Time array
            signal: Raw PPG signal

        Returns:
            Dictionary containing segmentation results
        """
        print("Starting PPG segmentation...")

        # Step 1: Preprocess
        print("  Preprocessing signal...")
        signal_processed = self.preprocess_signal(signal)

        # Step 2: Detect peaks
        print("  Detecting peaks...")
        peaks = self.detect_peaks(signal_processed)
        print(f"  Found {len(peaks)} peaks")

        if len(peaks) < 2:
            raise ValueError("Insufficient peaks detected. Check signal quality or parameters.")

        # Step 3: Detect valleys
        print("  Detecting valleys...")
        valleys = self.detect_valleys(signal_processed, peaks)
        print(f"  Found {len(valleys)} valleys")

        # Step 4: Match peaks and valleys, calculate quality
        print("  Calculating pulse quality...")
        pulses = []

        for i in range(len(valleys) - 1):
            onset_idx = valleys[i]
            next_onset_idx = valleys[i + 1]

            # Find peak between onsets
            peaks_between = peaks[(peaks > onset_idx) & (peaks < next_onset_idx)]

            if len(peaks_between) == 0:
                continue

            peak_idx = peaks_between[0]  # Use first peak

            # Calculate quality
            quality = self.calculate_pulse_quality(
                signal_processed,
                onset_idx,
                peak_idx,
                next_onset_idx
            )

            # Calculate metrics
            pulse_interval = (next_onset_idx - onset_idx) / self.sampling_rate
            heart_rate = 60.0 / pulse_interval if pulse_interval > 0 else 0

            amplitude = signal_processed[peak_idx] - signal_processed[onset_idx]

            pulse_data = {
                'pulse_number': i + 1,
                'onset_idx': int(onset_idx),
                'peak_idx': int(peak_idx),
                'next_onset_idx': int(next_onset_idx),
                'onset_time': float(time[onset_idx]),
                'peak_time': float(time[peak_idx]),
                'next_onset_time': float(time[next_onset_idx]),
                'pulse_interval': float(pulse_interval),
                'heart_rate': float(heart_rate),
                'amplitude': float(amplitude),
                'quality': float(quality),
                'valid': quality >= self.quality_threshold
            }

            pulses.append(pulse_data)

        # Step 5: Calculate summary statistics
        valid_pulses = [p for p in pulses if p['valid']]

        print(f"  Total pulses: {len(pulses)}")
        print(f"  Valid pulses: {len(valid_pulses)} ({len(valid_pulses)/len(pulses)*100:.1f}%)")

        if len(valid_pulses) > 0:
            valid_hrs = [p['heart_rate'] for p in valid_pulses]
            valid_intervals = [p['pulse_interval'] for p in valid_pulses]

            summary = {
                'total_pulses': len(pulses),
                'valid_pulses': len(valid_pulses),
                'validity_rate': len(valid_pulses) / len(pulses),
                'mean_heart_rate': np.mean(valid_hrs),
                'std_heart_rate': np.std(valid_hrs),
                'median_heart_rate': np.median(valid_hrs),
                'mean_interval': np.mean(valid_intervals),
                'std_interval': np.std(valid_intervals),
                'signal_duration': float(time[-1] - time[0])
            }
        else:
            summary = {
                'total_pulses': len(pulses),
                'valid_pulses': 0,
                'validity_rate': 0,
                'signal_duration': float(time[-1] - time[0])
            }

        result = {
            'pulses': pulses,
            'summary': summary,
            'preprocessed_signal': signal_processed.tolist(),
            'parameters': {
                'sampling_rate': self.sampling_rate,
                'min_heart_rate': self.min_heart_rate,
                'max_heart_rate': self.max_heart_rate,
                'quality_threshold': self.quality_threshold
            }
        }

        print("Segmentation complete!")
        return result

    def segment_from_file(self, csv_file: str,
                         output_file: Optional[str] = None) -> Dict:
        """
        Segment PPG signal from CSV file.

        Args:
            csv_file: Path to input CSV file
            output_file: Path to output JSON file (optional)

        Returns:
            Segmentation results dictionary
        """
        # Load data
        print(f"Loading data from: {csv_file}")
        time, signal = self.load_ppg_data(csv_file)

        # Estimate sampling rate if not provided
        if len(time) > 1:
            time_diffs = np.diff(time)
            median_interval = np.median(time_diffs)
            estimated_sr = 1.0 / median_interval if median_interval > 0 else self.sampling_rate

            if abs(estimated_sr - self.sampling_rate) > self.sampling_rate * 0.1:
                print(f"  Detected sampling rate: {estimated_sr:.2f} Hz (using {self.sampling_rate} Hz)")

        # Segment
        result = self.segment_pulses(time, signal)

        # Save to file
        if output_file:
            # Convert to serializable format
            output_data = {
                'input_file': csv_file,
                'pulses': result['pulses'],
                'summary': result['summary'],
                'parameters': result['parameters']
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to: {output_file}")

        # Also save CSV with pulse annotations
        if output_file:
            csv_output = output_file.replace('.json', '_pulses.csv')
            pulses_df = pd.DataFrame(result['pulses'])
            pulses_df.to_csv(csv_output, index=False)
            print(f"Pulse data saved to: {csv_output}")

        return result


def main():
    """Example usage of PPG segmenter."""
    import os

    print("PPG Segmentation Example")
    print("=" * 60)

    # Find PPG data files
    ppg_files = [f for f in os.listdir('.') if f.startswith('case_') and f.endswith('.csv')]

    if not ppg_files:
        print("No PPG data files found. Please run ppg_extractor.py first.")
        return

    print(f"Found {len(ppg_files)} PPG file(s)")

    # Use first file
    input_file = ppg_files[0]
    print(f"\nProcessing: {input_file}")

    # Create segmenter
    segmenter = PPGSegmenter(sampling_rate=500.0)  # Adjust based on your data

    # Segment
    output_file = input_file.replace('.csv', '_segmentation.json')
    result = segmenter.segment_from_file(input_file, output_file)

    # Print summary
    print("\nSegmentation Summary:")
    print(f"  Total pulses: {result['summary']['total_pulses']}")
    print(f"  Valid pulses: {result['summary']['valid_pulses']}")
    if result['summary']['valid_pulses'] > 0:
        print(f"  Mean HR: {result['summary']['mean_heart_rate']:.1f} bpm")
        print(f"  Std HR: {result['summary']['std_heart_rate']:.1f} bpm")
        print(f"  Mean interval: {result['summary']['mean_interval']:.3f} s")


if __name__ == "__main__":
    main()
