"""
PPG Peak Detection Pipeline
============================
Simplified pipeline that extracts PPG data, detects peaks, and visualizes results.

Usage:
    python ppg_peak_detection_pipeline.py --case-id 1
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from src.data_extraction.ppg_extractor import PPGExtractor
from src.data_extraction.ppg_segmentation import PPGSegmenter


def create_peak_visualization_html(case_id: int,
                                   csv_file: str,
                                   peaks: np.ndarray,
                                   time: np.ndarray,
                                   signal: np.ndarray,
                                   preprocessed_signal: np.ndarray,
                                   output_file: str,
                                   sampling_rate: float):
    """
    Create HTML report with peak detection visualization.

    Args:
        case_id: Case ID
        csv_file: Path to CSV file
        peaks: Array of peak indices
        time: Time array
        signal: Original signal array
        preprocessed_signal: Preprocessed signal array
        output_file: Output HTML file path
        sampling_rate: Sampling rate in Hz
    """
    # Create figures
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Original signal with peaks (first 30 seconds)
    duration = min(30, time[-1])
    mask = time <= duration

    axes[0].plot(time[mask], signal[mask], 'b-', linewidth=0.5, label='Original Signal', alpha=0.7)
    peak_mask = time[peaks] <= duration
    axes[0].plot(time[peaks][peak_mask], signal[peaks][peak_mask], 'ro',
                markersize=6, label=f'Detected Peaks ({np.sum(peak_mask)})')
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('PPG Amplitude', fontsize=12)
    axes[0].set_title(f'PPG Signal with Peak Detection - Case {case_id} (First 30s)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Preprocessed signal with peaks (first 30 seconds)
    axes[1].plot(time[mask], preprocessed_signal[mask], 'g-', linewidth=0.5, label='Preprocessed Signal', alpha=0.7)
    axes[1].plot(time[peaks][peak_mask], preprocessed_signal[peaks][peak_mask], 'ro',
                markersize=6, label=f'Detected Peaks ({np.sum(peak_mask)})')
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('PPG Amplitude (Preprocessed)', fontsize=12)
    axes[1].set_title(f'Preprocessed PPG Signal with Peak Detection - Case {case_id} (First 30s)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plot_file = output_file.replace('.html', '_peaks.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate statistics
    total_peaks = len(peaks)
    peak_times = time[peaks]
    if len(peak_times) > 1:
        intervals = np.diff(peak_times)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        mean_hr = 60.0 / mean_interval if mean_interval > 0 else 0
        std_hr = 60.0 * std_interval / (mean_interval ** 2) if mean_interval > 0 else 0
    else:
        mean_interval = 0
        std_interval = 0
        mean_hr = 0
        std_hr = 0

    # Create HTML report
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PPG Peak Detection - Case {case_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 5px 0;
            font-size: 16px;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        .stat-card .unit {{
            font-size: 14px;
            color: #666;
        }}
        .plot-section {{
            margin: 30px 0;
        }}
        .plot-section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 24px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        .info-section {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .info-section h3 {{
            color: #667eea;
            margin-top: 0;
        }}
        .info-section p {{
            margin: 8px 0;
            color: #555;
            line-height: 1.6;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 14px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PPG Peak Detection Analysis</h1>
            <p>Case ID: {case_id}</p>
            <p>VitalDB Dataset</p>
        </div>

        <div class="content">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Peaks</h3>
                    <div class="value">{total_peaks:,}</div>
                    <div class="unit">peaks detected</div>
                </div>

                <div class="stat-card">
                    <h3>Mean Heart Rate</h3>
                    <div class="value">{mean_hr:.1f}</div>
                    <div class="unit">± {std_hr:.1f} bpm</div>
                </div>

                <div class="stat-card">
                    <h3>Mean Interval</h3>
                    <div class="value">{mean_interval:.3f}</div>
                    <div class="unit">± {std_interval:.3f} seconds</div>
                </div>

                <div class="stat-card">
                    <h3>Sampling Rate</h3>
                    <div class="value">{sampling_rate:.0f}</div>
                    <div class="unit">Hz</div>
                </div>
            </div>

            <div class="info-section">
                <h3>Signal Information</h3>
                <p><strong>Duration:</strong> {time[-1]:.2f} seconds ({time[-1]/60:.2f} minutes)</p>
                <p><strong>Total Samples:</strong> {len(signal):,}</p>
                <p><strong>Data File:</strong> {os.path.basename(csv_file)}</p>
            </div>

            <div class="plot-section">
                <h2>Peak Detection Visualization</h2>
                <div class="plot-container">
                    <img src="{os.path.basename(plot_file)}" alt="Peak Detection Plot">
                </div>
            </div>

            <div class="info-section">
                <h3>Processing Steps</h3>
                <p><strong>1. Signal Extraction:</strong> PPG signal extracted from VitalDB</p>
                <p><strong>2. Preprocessing:</strong> DC removal, bandpass filtering (0.5-10 Hz), Savitzky-Golay smoothing</p>
                <p><strong>3. Peak Detection:</strong> Systolic peaks detected using scipy find_peaks with distance and prominence constraints</p>
                <p><strong>4. Visualization:</strong> Original and preprocessed signals shown with detected peaks overlaid</p>
            </div>
        </div>

        <div class="footer">
            <p>Generated with PPG Peak Detection Pipeline | VitalDB Dataset Analysis</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ HTML report saved: {output_file}")


def run_peak_detection_pipeline(case_id: int,
                                track_name: Optional[str] = None,
                                sampling_rate: float = None,
                                output_dir: str = './ppg_peak_analysis'):
    """
    Run simplified peak detection pipeline.

    Args:
        case_id: VitalDB case ID
        track_name: PPG track name (None for auto-select)
        sampling_rate: Sampling rate in Hz (auto-detect if None)
        output_dir: Output directory

    Returns:
        bool: True if successful
    """
    print("=" * 70)
    print("PPG PEAK DETECTION PIPELINE")
    print("=" * 70)
    print(f"Case ID: {case_id}")
    print(f"Track: {track_name if track_name else 'Auto-select'}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract PPG data
    print("\n[STEP 1/3] Extracting PPG data from VitalDB...")
    print("-" * 70)

    try:
        extractor = PPGExtractor()

        # Check available tracks
        available_tracks = extractor.get_available_ppg_tracks(case_id)

        if not available_tracks:
            print(f"ERROR: No PPG tracks found for case {case_id}")
            return False

        print(f"Available PPG tracks for case {case_id}:")
        for track in available_tracks:
            print(f"  - {track['tname']}")

        # Extract data
        if track_name and any(t['tname'] == track_name for t in available_tracks):
            print(f"\nAttempting to extract: {track_name}")
            try:
                result = extractor.extract_ppg(case_id, track_name, output_dir)
            except ValueError as e:
                print(f"WARNING: {track_name} failed: {e}")
                print("Trying alternative tracks...")
                result = extractor.extract_best_ppg_track(case_id, output_dir)
        else:
            if track_name:
                print(f"\nWARNING: Track '{track_name}' not available.")
            print("Finding best available PPG track...")
            result = extractor.extract_best_ppg_track(case_id, output_dir)

        csv_file = result['csv_file']

        print(f"\n✓ Extraction complete!")
        print(f"  File: {csv_file}")
        print(f"  Samples: {result['num_samples']:,}")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")

        # Auto-detect sampling rate
        if sampling_rate is None:
            sampling_rate = result.get('estimated_sampling_rate')
            if sampling_rate:
                print(f"  Detected sampling rate: {sampling_rate:.2f} Hz")
            else:
                sampling_rate = result.get('expected_sampling_rate', 500)
                print(f"  Using expected sampling rate: {sampling_rate} Hz")

    except Exception as e:
        print(f"ERROR in extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Peak detection
    print("\n[STEP 2/3] Detecting peaks in PPG signal...")
    print("-" * 70)

    try:
        segmenter = PPGSegmenter(sampling_rate=sampling_rate)

        # Load data
        time, signal = segmenter.load_ppg_data(csv_file)
        print(f"✓ Loaded signal: {len(signal):,} samples")

        # Preprocess
        preprocessed_signal = segmenter.preprocess_signal(signal)
        print(f"✓ Signal preprocessed")

        # Detect peaks
        peaks = segmenter.detect_peaks(preprocessed_signal)
        print(f"✓ Detected {len(peaks):,} peaks")

        if len(peaks) > 1:
            peak_times = time[peaks]
            intervals = np.diff(peak_times)
            mean_interval = np.mean(intervals)
            mean_hr = 60.0 / mean_interval if mean_interval > 0 else 0
            print(f"  Mean interval: {mean_interval:.3f} seconds")
            print(f"  Mean heart rate: {mean_hr:.1f} bpm")

    except Exception as e:
        print(f"ERROR in peak detection: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Generate HTML report
    print("\n[STEP 3/3] Generating HTML report...")
    print("-" * 70)

    try:
        html_file = os.path.join(output_dir, f"case_{case_id}_peaks_report.html")
        create_peak_visualization_html(
            case_id=case_id,
            csv_file=csv_file,
            peaks=peaks,
            time=time,
            signal=signal,
            preprocessed_signal=preprocessed_signal,
            output_file=html_file,
            sampling_rate=sampling_rate
        )

    except Exception as e:
        print(f"ERROR creating HTML report: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"✓ Case {case_id} analysis successful")
    print(f"\nOutput files:")
    print(f"  - PPG Data: {os.path.basename(csv_file)}")
    print(f"  - HTML Report: {os.path.basename(html_file)}")
    print(f"\nOpen the HTML report in your browser:")
    print(f"  file://{os.path.abspath(html_file)}")
    print("=" * 70)

    return True


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='PPG Peak Detection Pipeline for VitalDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze case 1
  python ppg_peak_detection_pipeline.py --case-id 1

  # With specific track
  python ppg_peak_detection_pipeline.py --case-id 1 --track "SNUADC/PLETH"

  # Custom output directory
  python ppg_peak_detection_pipeline.py --case-id 1 --output ./my_analysis
        """
    )

    parser.add_argument('--case-id', type=int, required=True,
                       help='VitalDB case ID to analyze')
    parser.add_argument('--track', type=str, default=None,
                       help='PPG track name (default: auto-select best track)')
    parser.add_argument('--sampling-rate', type=float, default=None,
                       help='Sampling rate in Hz (auto-detect if not specified)')
    parser.add_argument('--output', type=str, default='./ppg_peak_analysis',
                       help='Output directory (default: ./ppg_peak_analysis)')

    args = parser.parse_args()

    success = run_peak_detection_pipeline(
        case_id=args.case_id,
        track_name=args.track,
        sampling_rate=args.sampling_rate,
        output_dir=args.output
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
