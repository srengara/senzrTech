"""
Glucose Data Extractor for VitalDB Dataset
==========================================
Extracts glucose/blood sugar measurements from VitalDB cases.

Common glucose-related tracks:
- Laboratory/GLU (mg/dL) - Lab glucose measurements
- ISTAT/GLU (mg/dL) - Point-of-care glucose
- Solar8000/GLU (mg/dL) - Continuous glucose monitor
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
from src.utils.vitaldb_utility import VitalDBUtility

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlucoseExtractor:
    """Extract glucose measurements from VitalDB dataset."""

    # Common glucose track names
    GLUCOSE_TRACKS = [
        'Laboratory/GLU',      # Lab measurements
        'ISTAT/GLU',          # Point-of-care
        'Solar8000/GLU',      # Continuous monitor
        'BGA/GLU',            # Blood gas analyzer
        'Lab/Glucose',        # Alternative naming
        'Lab/GLU',           # Alternative naming
    ]

    def __init__(self):
        """Initialize glucose extractor."""
        self.util = VitalDBUtility()
        self._clinical_cache = None

    def get_available_glucose_tracks(self, case_id: int) -> List[Dict[str, any]]:
        """
        Get all available glucose tracks for a case.

        Args:
            case_id: VitalDB case ID

        Returns:
            List of dictionaries with track info
        """
        try:
            all_tracks = self.util.get_case_tracks(case_id)

            glucose_tracks = []
            for track in all_tracks:
                track_name = track.get('tname', '')
                # Check if track name contains glucose-related keywords
                if any(keyword in track_name.upper() for keyword in ['GLU', 'GLUCOSE', 'SUGAR', 'BG']):
                    glucose_tracks.append({
                        'tname': track_name,
                        'tid': track.get('tid'),
                        'type': self._identify_track_type(track_name)
                    })

            return glucose_tracks

        except Exception as e:
            logger.error(f"Error getting glucose tracks for case {case_id}: {e}")
            return []

    def _identify_track_type(self, track_name: str) -> str:
        """Identify the type of glucose measurement."""
        track_upper = track_name.upper()

        if 'LAB' in track_upper:
            return 'laboratory'
        elif 'ISTAT' in track_upper or 'POC' in track_upper:
            return 'point-of-care'
        elif 'SOLAR' in track_upper or 'CGM' in track_upper:
            return 'continuous'
        elif 'BGA' in track_upper:
            return 'blood-gas'
        else:
            return 'unknown'

    def get_clinical_glucose(self, case_id: int) -> Optional[float]:
        """
        Get preoperative glucose value from clinical information.

        Args:
            case_id: VitalDB case ID

        Returns:
            Glucose value in mg/dL, or None if not available
        """
        try:
            # Get case clinical information
            case_info = self.util.get_case_info(case_id)

            if not case_info:
                logger.warning(f"No clinical info found for case {case_id}")
                return None

            # Try different possible column names for preop glucose
            glucose_keys = ['preop_glucose', 'preop_glu', 'glucose', 'glu']

            for key in glucose_keys:
                if key in case_info:
                    try:
                        glucose_value = float(case_info[key])
                        if glucose_value > 0 and glucose_value < 1000:  # Sanity check
                            logger.info(f"Found preop glucose for case {case_id}: {glucose_value} mg/dL")
                            return glucose_value
                    except (ValueError, TypeError):
                        continue

            logger.warning(f"No preop glucose found in clinical info for case {case_id}")
            return None

        except Exception as e:
            logger.error(f"Error getting clinical glucose for case {case_id}: {e}")
            return None

    def extract_glucose_data(
        self,
        case_id: int,
        track_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Extract glucose data from a specific case.

        Args:
            case_id: VitalDB case ID
            track_name: Specific glucose track to extract (if None, tries common ones)

        Returns:
            DataFrame with columns: ['time', 'glucose_mg_dl']
        """
        try:
            # If no track specified, try to find one
            if track_name is None:
                available_tracks = self.get_available_glucose_tracks(case_id)
                if not available_tracks:
                    logger.warning(f"No glucose tracks found for case {case_id}")
                    return None

                # Use the first available track
                track_name = available_tracks[0]['name']
                logger.info(f"Using glucose track: {track_name}")

            # Load the track data
            data = self.util.load_track(case_id, track_name)

            if data is None or len(data) == 0:
                logger.warning(f"No data in glucose track {track_name} for case {case_id}")
                return None

            # Create DataFrame
            df = pd.DataFrame({
                'time': data['time'],
                'glucose_mg_dl': data['value']
            })

            # Remove NaN and invalid values
            df = df.dropna()
            df = df[df['glucose_mg_dl'] > 0]  # Glucose should be positive
            df = df[df['glucose_mg_dl'] < 1000]  # Remove unrealistic values

            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)

            logger.info(f"Extracted {len(df)} glucose measurements from case {case_id}")
            logger.info(f"Glucose range: {df['glucose_mg_dl'].min():.1f} - {df['glucose_mg_dl'].max():.1f} mg/dL")

            return df

        except Exception as e:
            logger.error(f"Error extracting glucose data: {e}")
            return None

    def match_glucose_to_ppg_windows(
        self,
        glucose_df: pd.DataFrame,
        ppg_times: np.ndarray,
        window_duration: float = 1.0,
        method: str = 'nearest'
    ) -> np.ndarray:
        """
        Match glucose measurements to PPG windows.

        Args:
            glucose_df: DataFrame with glucose data (time, glucose_mg_dl)
            ppg_times: Array of PPG window center times (in seconds)
            window_duration: Duration of each PPG window (seconds)
            method: Matching method
                - 'nearest': Use nearest glucose measurement
                - 'interpolate': Linear interpolation between measurements
                - 'last_known': Use last known value (forward fill)

        Returns:
            Array of glucose values for each PPG window
        """
        if glucose_df is None or len(glucose_df) == 0:
            logger.warning("No glucose data available for matching")
            return None

        glucose_values = np.full(len(ppg_times), np.nan)

        if method == 'nearest':
            # For each PPG window, find nearest glucose measurement
            for i, ppg_time in enumerate(ppg_times):
                time_diffs = np.abs(glucose_df['time'].values - ppg_time)
                nearest_idx = np.argmin(time_diffs)

                # Only use if within reasonable time window (e.g., 30 minutes)
                if time_diffs[nearest_idx] < 1800:  # 30 minutes
                    glucose_values[i] = glucose_df.iloc[nearest_idx]['glucose_mg_dl']

        elif method == 'interpolate':
            # Linear interpolation
            from scipy.interpolate import interp1d

            # Create interpolation function
            f = interp1d(
                glucose_df['time'].values,
                glucose_df['glucose_mg_dl'].values,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )

            glucose_values = f(ppg_times)

            # Set to NaN if outside measurement range
            min_time = glucose_df['time'].min()
            max_time = glucose_df['time'].max()
            glucose_values[(ppg_times < min_time) | (ppg_times > max_time)] = np.nan

        elif method == 'last_known':
            # Forward fill - use last known glucose value
            for i, ppg_time in enumerate(ppg_times):
                # Find all glucose measurements before this time
                prior_measurements = glucose_df[glucose_df['time'] <= ppg_time]

                if len(prior_measurements) > 0:
                    # Use the most recent measurement
                    glucose_values[i] = prior_measurements.iloc[-1]['glucose_mg_dl']

        else:
            raise ValueError(f"Unknown matching method: {method}")

        # Count valid matches
        valid_count = np.sum(~np.isnan(glucose_values))
        logger.info(f"Matched {valid_count}/{len(ppg_times)} PPG windows to glucose values")

        return glucose_values

    def create_training_dataset(
        self,
        case_id: int,
        ppg_track: str,
        glucose_track: Optional[str] = None,
        output_dir: str = './training_data',
        matching_method: str = 'interpolate'
    ) -> Tuple[str, str]:
        """
        Create complete training dataset with PPG windows and glucose labels.

        Args:
            case_id: VitalDB case ID
            ppg_track: PPG track name
            glucose_track: Glucose track name (auto-detect if None)
            output_dir: Directory to save output files
            matching_method: Method to match glucose to PPG windows

        Returns:
            Tuple of (ppg_file_path, glucose_file_path)
        """
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.data_extraction.ppg_extractor import PPGExtractor
        from src.data_extraction.ppg_segmentation import PPGSegmenter
        from src.data_extraction.peak_detection import ppg_peak_detection_pipeline_with_template

        try:
            # 1. Extract PPG data
            logger.info(f"Extracting PPG data from case {case_id}, track {ppg_track}")
            ppg_extractor = PPGExtractor()
            ppg_data = ppg_extractor.extract_ppg(case_id, ppg_track)

            if ppg_data is None:
                raise ValueError("Failed to extract PPG data")

            # 2. Preprocess PPG
            logger.info("Preprocessing PPG signal...")
            segmenter = PPGSegmenter(sampling_rate=ppg_data['sampling_rate'])
            preprocessed_signal = segmenter.preprocess_signal(ppg_data['signal'])

            # 3. Detect peaks and extract windows
            logger.info("Detecting peaks and filtering windows...")
            peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
                ppg_signal=preprocessed_signal,
                fs=ppg_data['sampling_rate'],
                window_duration=1.0,
                similarity_threshold=0.85
            )

            if len(filtered_windows) == 0:
                raise ValueError("No valid PPG windows extracted")

            logger.info(f"Extracted {len(filtered_windows)} filtered PPG windows")

            # 4. Extract glucose data
            logger.info(f"Extracting glucose data from case {case_id}")
            glucose_df = self.extract_glucose_data(case_id, glucose_track)

            if glucose_df is None:
                raise ValueError("Failed to extract glucose data")

            # 5. Calculate PPG window times
            ppg_window_times = peaks / ppg_data['sampling_rate']

            # Only use windows where we have filtered data
            filtered_indices = [i for i, w in enumerate(all_windows) if any((w == fw).all() for fw in filtered_windows)]
            ppg_window_times = ppg_window_times[filtered_indices]

            # 6. Match glucose to PPG windows
            logger.info(f"Matching glucose values to PPG windows using method: {matching_method}")
            glucose_values = self.match_glucose_to_ppg_windows(
                glucose_df,
                ppg_window_times,
                method=matching_method
            )

            # 7. Remove windows without glucose values
            valid_mask = ~np.isnan(glucose_values)
            filtered_windows = [w for i, w in enumerate(filtered_windows) if valid_mask[i]]
            glucose_values = glucose_values[valid_mask]

            logger.info(f"Final dataset: {len(filtered_windows)} PPG windows with glucose labels")

            # 8. Save PPG windows to CSV (detailed format)
            os.makedirs(output_dir, exist_ok=True)

            ppg_file = os.path.join(output_dir, f'case_{case_id}_ppg_windows.csv')
            ppg_rows = []
            for window_idx, window in enumerate(filtered_windows):
                for sample_idx, amplitude in enumerate(window):
                    ppg_rows.append({
                        'window_index': window_idx,
                        'sample_index': sample_idx,
                        'amplitude': amplitude
                    })

            ppg_df = pd.DataFrame(ppg_rows)
            ppg_df.to_csv(ppg_file, index=False)
            logger.info(f"Saved PPG windows to: {ppg_file}")

            # 9. Save glucose labels to CSV
            glucose_file = os.path.join(output_dir, f'case_{case_id}_glucose_labels.csv')
            glucose_label_df = pd.DataFrame({
                'window_index': range(len(glucose_values)),
                'glucose_mg_dl': glucose_values
            })
            glucose_label_df.to_csv(glucose_file, index=False)
            logger.info(f"Saved glucose labels to: {glucose_file}")

            # Print statistics
            logger.info("\n" + "="*60)
            logger.info("Dataset Statistics")
            logger.info("="*60)
            logger.info(f"Number of windows: {len(glucose_values)}")
            logger.info(f"Glucose mean: {np.mean(glucose_values):.1f} mg/dL")
            logger.info(f"Glucose std: {np.std(glucose_values):.1f} mg/dL")
            logger.info(f"Glucose min: {np.min(glucose_values):.1f} mg/dL")
            logger.info(f"Glucose max: {np.max(glucose_values):.1f} mg/dL")
            logger.info("="*60)

            return ppg_file, glucose_file

        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            raise


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract glucose data from VitalDB')
    parser.add_argument('--case_id', type=int, required=True, help='VitalDB case ID')
    parser.add_argument('--ppg_track', type=str, default='SNUADC/PLETH', help='PPG track name')
    parser.add_argument('--glucose_track', type=str, default=None, help='Glucose track name (auto-detect if not specified)')
    parser.add_argument('--output_dir', type=str, default='./training_data', help='Output directory')
    parser.add_argument('--matching_method', type=str, default='interpolate',
                       choices=['nearest', 'interpolate', 'last_known'],
                       help='Method to match glucose to PPG windows')

    args = parser.parse_args()

    # Create extractor
    extractor = GlucoseExtractor()

    # Check available glucose tracks
    print(f"\nChecking glucose tracks for case {args.case_id}...")
    glucose_tracks = extractor.get_available_glucose_tracks(args.case_id)

    if glucose_tracks:
        print(f"Found {len(glucose_tracks)} glucose tracks:")
        for track in glucose_tracks:
            print(f"  - {track['name']} ({track['type']})")
    else:
        print("No glucose tracks found!")
        return

    # Create training dataset
    print(f"\nCreating training dataset...")
    try:
        ppg_file, glucose_file = extractor.create_training_dataset(
            case_id=args.case_id,
            ppg_track=args.ppg_track,
            glucose_track=args.glucose_track,
            output_dir=args.output_dir,
            matching_method=args.matching_method
        )

        print(f"\n✓ Training dataset created successfully!")
        print(f"  PPG windows: {ppg_file}")
        print(f"  Glucose labels: {glucose_file}")

    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == '__main__':
    main()
