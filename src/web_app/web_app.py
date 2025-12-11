"""
VitalDB PPG Analysis Web Interface
===================================
Interactive web application for PPG data extraction, cleansing, and peak detection.

Usage:
    python web_app.py
    Then open http://localhost:5000 in your browser
"""

import os
import json
import base64
import logging
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data_extraction.ppg_extractor import PPGExtractor
from src.data_extraction.ppg_segmentation import PPGSegmenter
from src.data_extraction.peak_detection import (
    ppg_peak_detection_pipeline,
    ppg_peak_detection_pipeline_with_template,
    compute_template,
    filter_windows_by_similarity,
    cosine_similarity
)
from src.data_extraction.glucose_extractor import GlucoseExtractor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vitaldb-ppg-analysis'
app.config['UPLOAD_FOLDER'] = './data/web_app_data'

# Ensure data directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global cache for session data
session_data = {}


def create_plot(time, signal, title, peaks=None, max_duration=30):
    """Create a matplotlib plot and return as base64 image."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Limit to max_duration seconds
    mask = time <= max_duration

    ax.plot(time[mask], signal[mask], 'b-', linewidth=0.8, alpha=0.7)

    if peaks is not None:
        peak_mask = time[peaks] <= max_duration
        ax.plot(time[peaks][peak_mask], signal[peaks][peak_mask],
                'ro', markersize=6, label=f'Peaks ({np.sum(peak_mask)})')
        ax.legend()

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('PPG Amplitude', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return f"data:image/png;base64,{image_base64}"


def create_template_plot(template, sampling_rate):
    """Create a plot of the template waveform."""
    if template.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    # Create time axis for template
    time_template = np.arange(len(template)) / sampling_rate

    ax.plot(time_template, template, 'g-', linewidth=2, label='Template')
    ax.fill_between(time_template, template, alpha=0.3, color='green')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title('Computed PPG Template (Average Beat)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return f"data:image/png;base64,{image_base64}"


def create_windows_comparison_plot(all_windows, filtered_windows, template, sampling_rate):
    """Create a plot showing all windows vs filtered windows."""
    if len(all_windows) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Plot 1: All windows
    ax1 = axes[0]
    for i, window in enumerate(all_windows[:20]):  # Show first 20
        time_window = np.arange(len(window)) / sampling_rate
        ax1.plot(time_window, window, 'b-', alpha=0.3, linewidth=0.8)

    if template.size > 0:
        time_template = np.arange(len(template)) / sampling_rate
        ax1.plot(time_template, template, 'r-', linewidth=2, label='Template', alpha=0.8)

    ax1.set_xlabel('Time (seconds)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title(f'All Extracted Windows (n={len(all_windows)})', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Filtered windows
    ax2 = axes[1]
    for i, window in enumerate(filtered_windows[:20]):  # Show first 20
        time_window = np.arange(len(window)) / sampling_rate
        ax2.plot(time_window, window, 'g-', alpha=0.4, linewidth=0.8)

    if template.size > 0:
        time_template = np.arange(len(template)) / sampling_rate
        ax2.plot(time_template, template, 'r-', linewidth=2, label='Template', alpha=0.8)

    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title(f'Filtered Windows (n={len(filtered_windows)})', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/get_tracks', methods=['POST'])
def get_tracks():
    """Get available PPG tracks for a case."""
    try:
        case_id = int(request.json.get('case_id'))

        extractor = PPGExtractor()
        tracks = extractor.get_available_ppg_tracks(case_id)

        if not tracks:
            return jsonify({'error': 'No PPG tracks found for this case'}), 404

        # Format tracks for dropdown
        track_list = []
        for track in tracks:
            track_name = track['tname']
            expected_rate = extractor.PPG_TRACKS.get(track_name, 'Unknown')
            track_list.append({
                'name': track_name,
                'rate': expected_rate
            })

        return jsonify({'tracks': track_list})

    except ValueError:
        return jsonify({'error': 'Invalid case ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_data', methods=['POST'])
def download_data():
    """Download raw PPG data from VitalDB."""
    try:
        case_id = int(request.json.get('case_id'))
        track_name = request.json.get('track_name')

        if not track_name:
            return jsonify({'error': 'Track name is required'}), 400

        # Create session directory
        session_id = f"case_{case_id}_{track_name.replace('/', '_')}"
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Extract RAW data (no modifications)
        extractor = PPGExtractor()
        result = extractor.extract_ppg_raw(case_id, track_name, session_dir)

        # Load raw data
        df = pd.read_csv(result['csv_file'])

        # Check for issues
        total_rows = len(df)
        nan_time = df['time'].isna().sum()
        nan_signal = df['ppg'].isna().sum()

        has_issues = (nan_time > 0 or nan_signal > 0)

        # Store in session
        session_data[session_id] = {
            'case_id': case_id,
            'track_name': track_name,
            'csv_file': result['csv_file'],
            'metadata': result,
            'has_issues': has_issues
        }

        # Create plot
        time = df['time'].values
        signal = df['ppg'].values

        # For plotting, use first valid values
        valid_mask = ~(np.isnan(time) | np.isnan(signal))
        if np.sum(valid_mask) > 0:
            plot_image = create_plot(
                time[valid_mask],
                signal[valid_mask],
                f'Raw PPG Signal - Case {case_id} - {track_name} (First 30s)'
            )
        else:
            plot_image = None

        # Get first 100 rows for preview
        # Replace NaN with None (null in JSON) for proper serialization
        preview_data = df.head(100).replace({np.nan: None}).to_dict('records')

        # Convert numpy/pandas types to Python native types for JSON serialization
        sampling_rate = result.get('estimated_sampling_rate', result.get('expected_sampling_rate', 500))
        if sampling_rate is not None:
            sampling_rate = float(sampling_rate)

        duration = result.get('duration_seconds', 0)
        if duration is not None:
            duration = float(duration)

        return jsonify({
            'session_id': session_id,
            'total_rows': int(total_rows),
            'nan_time': int(nan_time),
            'nan_signal': int(nan_signal),
            'has_issues': bool(has_issues),
            'sampling_rate': sampling_rate,
            'duration': duration,
            'preview_data': preview_data,
            'plot_image': plot_image
        })

    except ValueError as e:
        return jsonify({'error': f'Data extraction failed: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanse_data', methods=['POST'])
def cleanse_data():
    """Cleanse PPG data (handle NaN values)."""
    try:
        session_id = request.json.get('session_id')

        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        session = session_data[session_id]
        csv_file = session['csv_file']

        # Load data
        df = pd.read_csv(csv_file)

        # Get original stats
        original_rows = len(df)

        # Cleansing logic (from ppg_extractor.py)
        time_col = 'time'
        signal_col = 'ppg'

        # Remove NaN from signal
        df = df.dropna(subset=[signal_col])

        # Handle time column
        if df[time_col].isna().all():
            # Create synthetic time values
            sampling_rate = session['metadata'].get('expected_sampling_rate', 500)
            df[time_col] = np.arange(len(df)) / sampling_rate
        else:
            # Remove rows with NaN time
            df = df.dropna(subset=[time_col])

        # Save cleansed data
        cleansed_file = csv_file.replace('.csv', '_cleansed.csv')
        df.to_csv(cleansed_file, index=False)

        # Update session
        session['cleansed_file'] = cleansed_file
        session['cleansed_rows'] = len(df)

        # Create plot
        time = df['time'].values
        signal = df['ppg'].values

        plot_image = create_plot(
            time,
            signal,
            f'Cleansed PPG Signal - Case {session["case_id"]} (First 30s)'
        )

        # Get preview
        # Replace NaN with None for JSON serialization
        preview_data = df.head(100).replace({np.nan: None}).to_dict('records')

        return jsonify({
            'original_rows': original_rows,
            'cleansed_rows': len(df),
            'removed_rows': original_rows - len(df),
            'preview_data': preview_data,
            'plot_image': plot_image
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect_peaks', methods=['POST'])
def detect_peaks():
    """Detect peaks in cleansed PPG data."""
    try:
        session_id = request.json.get('session_id')
        height_multiplier = request.json.get('height_multiplier', 0.3)  # Default 0.3
        distance_multiplier = request.json.get('distance_multiplier', 0.8)  # Default 0.8

        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        session = session_data[session_id]

        # Use cleansed file if available, otherwise raw
        csv_file = session.get('cleansed_file', session['csv_file'])

        # Load data
        df = pd.read_csv(csv_file)
        time = df['time'].values
        signal = df['ppg'].values

        # Get sampling rate
        sampling_rate = session['metadata'].get('estimated_sampling_rate')
        if sampling_rate is None:
            sampling_rate = session['metadata'].get('expected_sampling_rate', 500)

        # Preprocess signal for better peak detection
        segmenter = PPGSegmenter(sampling_rate=sampling_rate)
        preprocessed_signal = segmenter.preprocess_signal(signal)

        # Use peak_detection.py algorithm
        # Calculate adaptive height threshold based on signal statistics and user input
        signal_mean = np.mean(preprocessed_signal)
        signal_std = np.std(preprocessed_signal)
        height_threshold = signal_mean + float(height_multiplier) * signal_std

        # Calculate distance threshold based on user input
        distance_threshold = float(distance_multiplier) * float(sampling_rate)

        # Run template-based peak detection pipeline
        peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
            ppg_signal=preprocessed_signal,
            fs=float(sampling_rate),
            window_duration=1.0,
            height_threshold=float(height_threshold),
            distance_threshold=distance_threshold,
            similarity_threshold=0.85
        )

        # Convert peaks list to numpy array for indexing
        peaks_array = np.array(peaks, dtype=int)

        # Calculate filtering statistics
        num_all_windows = len(all_windows)
        num_filtered_windows = len(filtered_windows)
        filtering_rate = (num_filtered_windows / num_all_windows * 100) if num_all_windows > 0 else 0

        # Calculate statistics
        total_peaks = len(peaks_array)
        peak_times = time[peaks_array]

        if len(peak_times) > 1:
            intervals = np.diff(peak_times)
            mean_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))
            mean_hr = 60.0 / mean_interval if mean_interval > 0 else 0
            std_hr = 60.0 * std_interval / (mean_interval ** 2) if mean_interval > 0 else 0
        else:
            mean_interval = 0
            std_interval = 0
            mean_hr = 0
            std_hr = 0

        # Create plots (use peaks_array for indexing)
        plot_original = create_plot(
            time, signal,
            f'Original Signal with Peaks - Case {session["case_id"]}',
            peaks=peaks_array
        )

        plot_preprocessed = create_plot(
            time, preprocessed_signal,
            f'Preprocessed Signal with Peaks - Case {session["case_id"]}',
            peaks=peaks_array
        )

        # Create peaks dataframe (use peaks_array for indexing)
        peaks_df = pd.DataFrame({
            'peak_index': peaks_array,
            'time': time[peaks_array],
            'amplitude_original': signal[peaks_array],
            'amplitude_preprocessed': preprocessed_signal[peaks_array]
        })

        # Save peaks data
        peaks_file = csv_file.replace('.csv', '_peaks.csv')
        peaks_df.to_csv(peaks_file, index=False)

        # Update session (peaks is already a list, no need for .tolist())
        session['peaks_file'] = peaks_file
        session['peaks'] = peaks  # Already a list from peak_detection.py
        session['total_peaks'] = total_peaks

        # Save filtered windows data
        if len(filtered_windows) > 0:
            # Create a dataframe with window information
            windows_data = []
            for i, window in enumerate(filtered_windows):
                similarity = cosine_similarity(window, template) if template.size > 0 else 0
                windows_data.append({
                    'window_index': i,
                    'peak_index': peaks[i] if i < len(peaks) else None,
                    'window_length': len(window),
                    'similarity_score': round(float(similarity), 4),
                    'window_mean': round(float(np.mean(window)), 4),
                    'window_std': round(float(np.std(window)), 4),
                    'window_min': round(float(np.min(window)), 4),
                    'window_max': round(float(np.max(window)), 4)
                })

            windows_df = pd.DataFrame(windows_data)
            windows_file = csv_file.replace('.csv', '_filtered_windows.csv')
            windows_df.to_csv(windows_file, index=False)
            session['filtered_windows_file'] = windows_file

            # Also save detailed window samples
            detailed_windows_file = csv_file.replace('.csv', '_filtered_windows_detailed.csv')
            detailed_data = []
            for i, window in enumerate(filtered_windows):
                for sample_idx, value in enumerate(window):
                    detailed_data.append({
                        'window_index': i,
                        'sample_index': sample_idx,
                        'amplitude': round(float(value), 4)
                    })
            detailed_windows_df = pd.DataFrame(detailed_data)
            detailed_windows_df.to_csv(detailed_windows_file, index=False)
            session['filtered_windows_detailed_file'] = detailed_windows_file

        # Create template plot
        plot_template = create_template_plot(template, sampling_rate)

        # Create windows comparison plot
        plot_windows_comparison = create_windows_comparison_plot(
            all_windows, filtered_windows, template, sampling_rate
        )

        # Calculate similarity scores for filtered windows
        similarity_scores = []
        if template.size > 0 and len(filtered_windows) > 0:
            for window in filtered_windows[:100]:  # First 100
                sim = cosine_similarity(window, template)
                similarity_scores.append(round(float(sim), 4))

        # Get preview
        # Replace NaN with None for JSON serialization
        preview_data = peaks_df.head(100).replace({np.nan: None}).to_dict('records')

        return jsonify({
            'total_peaks': total_peaks,
            'mean_hr': round(mean_hr, 2),
            'std_hr': round(std_hr, 2),
            'mean_interval': round(mean_interval, 4),
            'std_interval': round(std_interval, 4),
            'height_threshold': round(height_threshold, 4),
            'distance_threshold': round(distance_threshold, 2),
            'height_multiplier': height_multiplier,
            'distance_multiplier': distance_multiplier,
            'num_all_windows': num_all_windows,
            'num_filtered_windows': num_filtered_windows,
            'filtering_rate': round(filtering_rate, 2),
            'template_length': int(len(template)) if template.size > 0 else 0,
            'mean_similarity': round(float(np.mean(similarity_scores)), 4) if similarity_scores else 0,
            'min_similarity': round(float(np.min(similarity_scores)), 4) if similarity_scores else 0,
            'preview_data': preview_data,
            'plot_original': plot_original,
            'plot_preprocessed': plot_preprocessed,
            'plot_template': plot_template,
            'plot_windows_comparison': plot_windows_comparison
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract_glucose', methods=['POST'])
def extract_glucose():
    """Extract glucose data from VitalDB for the current case."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        matching_method = data.get('matching_method', 'interpolate')

        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400

        session = session_data[session_id]

        # Check if peaks exist
        if 'peaks' not in session:
            return jsonify({'error': 'No peaks available. Run peak detection first.'}), 400

        case_id = session.get('case_id')
        if not case_id:
            return jsonify({'error': 'Case ID not found in session'}), 400

        # Initialize glucose extractor
        glucose_extractor = GlucoseExtractor()

        # Try to get glucose from tracks first
        glucose_tracks = glucose_extractor.get_available_glucose_tracks(case_id)
        glucose_df = None
        glucose_source = None

        if glucose_tracks:
            # Extract glucose data from track (use first available track)
            glucose_track = glucose_tracks[0]['tname']
            glucose_df = glucose_extractor.extract_glucose_data(case_id, glucose_track)
            glucose_source = f'track:{glucose_track}'

        # If no track data, try clinical preop glucose
        if glucose_df is None or len(glucose_df) == 0:
            logger.info(f"No glucose track data available for case {case_id}, trying clinical glucose...")
            clinical_glucose = glucose_extractor.get_clinical_glucose(case_id)

            if clinical_glucose is None:
                return jsonify({
                    'error': f'No glucose data found for case {case_id}',
                    'suggestion': 'No glucose tracks or preop_glucose in clinical data. Try manual entry or use a different case.',
                    'checked': {
                        'glucose_tracks': len(glucose_tracks) if glucose_tracks else 0,
                        'clinical_glucose': 'not available'
                    }
                }), 404

            # Use clinical glucose as a constant value for all windows
            # This is a single preoperative measurement, so it's the same for all windows
            glucose_source = 'clinical:preop_glucose'
            use_constant_glucose = True
        else:
            use_constant_glucose = False
            clinical_glucose = None

        # Calculate PPG window times from peaks
        peaks = np.array(session['peaks'])

        # Get sampling rate from metadata
        sampling_rate = session['metadata'].get('estimated_sampling_rate')
        if sampling_rate is None:
            sampling_rate = session['metadata'].get('expected_sampling_rate', 500)

        # Load time data to get actual peak times
        csv_file = session.get('cleansed_file', session['csv_file'])
        df = pd.read_csv(csv_file)
        time = df['time'].values

        ppg_window_times = time[peaks]

        # Match glucose to PPG windows
        if use_constant_glucose:
            # Use the same clinical glucose value for all windows
            glucose_values = np.full(len(ppg_window_times), clinical_glucose)
            logger.info(f"Using constant preop glucose ({clinical_glucose} mg/dL) for all {len(ppg_window_times)} windows")
        else:
            # Match time-varying glucose from tracks
            glucose_values = glucose_extractor.match_glucose_to_ppg_windows(
                glucose_df,
                ppg_window_times,
                method=matching_method
            )

        # Filter to keep only windows with valid glucose values
        valid_mask = ~np.isnan(glucose_values)
        valid_count = np.sum(valid_mask)

        if valid_count == 0:
            return jsonify({
                'error': 'Could not match any glucose values to PPG windows',
                'glucose_measurements': len(glucose_df) if glucose_df is not None else 0,
                'ppg_windows': len(ppg_window_times),
                'suggestion': 'Glucose measurements may not overlap with PPG recording time'
            }), 404

        # Keep only valid glucose values and their corresponding window indices
        valid_indices = np.where(valid_mask)[0]
        valid_glucose_values = glucose_values[valid_mask]

        # Create glucose labels dataframe with ONLY valid windows
        # Format: [window_index, glucose_mg_dl]
        glucose_df_out = pd.DataFrame({
            'window_index': valid_indices,
            'glucose_mg_dl': valid_glucose_values
        })

        # Save glucose labels CSV
        session_dir = os.path.dirname(session['csv_file'])
        glucose_file = os.path.join(session_dir, 'glucose_labels.csv')
        glucose_df_out.to_csv(glucose_file, index=False)

        # Also need to save FILTERED ppg_windows.csv to match the valid glucose indices
        # Load the detailed windows file
        detailed_windows_file = session.get('filtered_windows_detailed_file')
        if detailed_windows_file and os.path.exists(detailed_windows_file):
            ppg_windows_df = pd.read_csv(detailed_windows_file)

            # Keep only windows that have valid glucose
            filtered_ppg_df = ppg_windows_df[ppg_windows_df['window_index'].isin(valid_indices)]

            # Re-index windows to be contiguous (0, 1, 2, ...)
            window_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
            filtered_ppg_df['window_index'] = filtered_ppg_df['window_index'].map(window_mapping)

            # Save filtered PPG windows
            ppg_windows_file = os.path.join(session_dir, 'ppg_windows.csv')
            filtered_ppg_df.to_csv(ppg_windows_file, index=False)
            session['ppg_windows_file'] = ppg_windows_file

            # Update glucose labels with re-indexed windows
            glucose_df_out['window_index'] = range(len(valid_glucose_values))
            glucose_df_out.to_csv(glucose_file, index=False)

        # Store in session
        session['glucose_labels'] = valid_glucose_values.tolist()
        session['glucose_labels_file'] = glucose_file
        session['glucose_source'] = glucose_source
        session['glucose_matching_method'] = matching_method if not use_constant_glucose else 'constant'
        session['valid_window_indices'] = valid_indices.tolist()

        # Create appropriate message based on source
        if use_constant_glucose:
            message = f'Glucose data extracted from clinical preop_glucose ({clinical_glucose} mg/dL)'
        else:
            message = f'Glucose data extracted from {glucose_source}'

        return jsonify({
            'success': True,
            'message': message,
            'glucose_source': glucose_source,
            'matching_method': matching_method if not use_constant_glucose else 'constant',
            'total_windows': len(ppg_window_times),
            'num_valid': int(valid_count),
            'num_removed': int(len(glucose_values) - valid_count),
            'file_path': glucose_file,
            'ppg_windows_file': session.get('ppg_windows_file', 'N/A'),
            'stats': {
                'mean': float(np.mean(valid_glucose_values)),
                'std': float(np.std(valid_glucose_values)),
                'min': float(np.min(valid_glucose_values)),
                'max': float(np.max(valid_glucose_values))
            },
            'available_tracks': [{'name': t['tname'], 'rate': t.get('rate', 'Unknown')} for t in glucose_tracks] if glucose_tracks else []
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_glucose_labels', methods=['POST'])
def save_glucose_labels():
    """Save manually entered glucose labels for training purposes."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        glucose_input = data.get('glucose_value')  # Single glucose value or list

        if not session_id or session_id not in session_data:
            return jsonify({'error': 'Invalid session'}), 400

        session = session_data[session_id]

        # Check if peaks exist
        if 'peaks' not in session:
            return jsonify({'error': 'No peaks available. Run peak detection first.'}), 400

        num_windows = session['total_peaks']

        # Handle both single value and list inputs
        if glucose_input is None:
            return jsonify({'error': 'glucose_value is required'}), 400

        # Convert single value to array of N values
        if isinstance(glucose_input, (int, float)):
            # Single value - replicate across all windows
            glucose_value = float(glucose_input)
            glucose_labels = [glucose_value] * num_windows
            logger.info(f"Replicating manual glucose value ({glucose_value} mg/dL) across {num_windows} windows")
        elif isinstance(glucose_input, list):
            # List provided - validate length
            if len(glucose_input) != num_windows:
                return jsonify({
                    'error': f'Number of glucose labels ({len(glucose_input)}) must match number of peaks ({num_windows})'
                }), 400
            glucose_labels = [float(g) for g in glucose_input]
        else:
            return jsonify({'error': 'glucose_value must be a number or list'}), 400

        # Create glucose labels dataframe
        # Format: [window_index, glucose_mg_dl] - simple 2 column format
        glucose_df = pd.DataFrame({
            'window_index': range(num_windows),
            'glucose_mg_dl': glucose_labels
        })

        # Save to CSV in session directory
        session_dir = os.path.dirname(session['csv_file'])
        glucose_file = os.path.join(session_dir, 'glucose_labels.csv')
        glucose_df.to_csv(glucose_file, index=False)

        # Also save the ppg_windows.csv from the detailed file
        detailed_windows_file = session.get('filtered_windows_detailed_file')
        if detailed_windows_file and os.path.exists(detailed_windows_file):
            ppg_windows_file = os.path.join(session_dir, 'ppg_windows.csv')
            ppg_windows_df = pd.read_csv(detailed_windows_file)
            ppg_windows_df.to_csv(ppg_windows_file, index=False)
            session['ppg_windows_file'] = ppg_windows_file

        # Store in session
        session['glucose_labels'] = glucose_labels
        session['glucose_labels_file'] = glucose_file
        session['glucose_source'] = 'manual'

        return jsonify({
            'success': True,
            'message': f'Glucose labels saved successfully',
            'source': 'manual',
            'num_labels': num_windows,
            'num_valid': num_windows,
            'num_removed': 0,
            'file_path': glucose_file,
            'ppg_windows_file': session.get('ppg_windows_file', 'N/A'),
            'stats': {
                'mean': float(np.mean(glucose_labels)),
                'std': float(np.std(glucose_labels)),
                'min': float(np.min(glucose_labels)),
                'max': float(np.max(glucose_labels))
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_csv/<session_id>/<file_type>')
def download_csv(session_id, file_type):
    """Download CSV file."""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404

        session = session_data[session_id]

        if file_type == 'raw':
            file_path = session['csv_file']
        elif file_type == 'cleansed':
            file_path = session.get('cleansed_file')
        elif file_type == 'peaks':
            file_path = session.get('peaks_file')
        elif file_type == 'filtered_windows':
            file_path = session.get('filtered_windows_file')
        elif file_type == 'filtered_windows_detailed':
            file_path = session.get('filtered_windows_detailed_file')
        elif file_type == 'ppg_windows':
            file_path = session.get('ppg_windows_file')
        elif file_type == 'glucose_labels':
            file_path = session.get('glucose_labels_file')
        else:
            return jsonify({'error': 'Invalid file type'}), 400

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("VitalDB PPG Analysis Web Interface")
    print("=" * 70)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)
