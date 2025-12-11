"""
PPG Signal Extractor for VitalDB Dataset
=========================================
Extracts PPG (Photoplethysmography) signals from VitalDB cases.

Common PPG tracks:
- SNUADC/PLETH - 500 Hz sampling rate
- Solar8000/PLETH - 62.5 Hz sampling rate
- Primus/PLETH - 100 Hz sampling rate
- BIS/PLETH - Variable sampling rate
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging
from src.utils.vitaldb_utility import VitalDBUtility

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPGExtractor:
    """Extract PPG signals from VitalDB dataset."""

    # Common PPG track names and their typical sampling rates
    PPG_TRACKS = {
        'SNUADC/PLETH': 500,      # Hz
        'Solar8000/PLETH': 62.5,   # Hz
        'Primus/PLETH': 100,       # Hz
        'BIS/PLETH': None,         # Variable
        'Solar8000/PLETH_SPO2': 1  # SpO2 values (not waveform)
    }

    def __init__(self):
        """Initialize PPG extractor."""
        self.util = VitalDBUtility()

    def get_available_ppg_tracks(self, case_id: int) -> List[Dict[str, str]]:
        """
        Get all available PPG tracks for a case.

        Args:
            case_id: VitalDB case ID

        Returns:
            List of dictionaries with track name and track ID
        """
        logger.info(f"Finding PPG tracks for case {case_id}...")

        # Get all tracks for the case
        all_tracks = self.util.get_case_tracks(case_id)

        # Filter PPG tracks
        ppg_tracks = [
            track for track in all_tracks
            if 'PLETH' in track['tname'].upper()
        ]

        logger.info(f"Found {len(ppg_tracks)} PPG tracks for case {case_id}")
        for track in ppg_tracks:
            sampling_rate = self.PPG_TRACKS.get(track['tname'], 'Unknown')
            logger.info(f"  - {track['tname']} (expected rate: {sampling_rate} Hz)")

        return ppg_tracks

    def extract_ppg_raw(self, case_id: int, track_name: str,
                        output_dir: str = './ppg_data') -> Dict:
        """
        Extract RAW PPG data WITHOUT any modifications or cleansing.
        Returns data exactly as it comes from VitalDB API with NaN values intact.

        Args:
            case_id: VitalDB case ID
            track_name: PPG track name (e.g., 'SNUADC/PLETH')
            output_dir: Directory to save extracted data

        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting RAW PPG signal: Case {case_id}, Track: {track_name}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get case tracks
        tracks = self.util.get_case_tracks(case_id)

        # Find matching track
        matching_track = None
        for track in tracks:
            if track['tname'] == track_name:
                matching_track = track
                break

        if not matching_track:
            raise ValueError(f"Track '{track_name}' not found for case {case_id}")

        # Download track data
        track_id = matching_track['tid']
        logger.info(f"Downloading data for track ID: {track_id}")
        raw_data = self.util.download_track_data(track_id)

        if not raw_data:
            raise ValueError(f"No data available for track {track_name}")

        # Convert to DataFrame
        df = pd.DataFrame(raw_data)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Get time and signal columns
        time_col = 'Time'
        signal_col = track_name

        if time_col not in df.columns:
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                time_col = time_cols[0]
            else:
                raise ValueError(f"Time column not found in data")

        if signal_col not in df.columns:
            signal_cols = [col for col in df.columns if col != time_col]
            if signal_cols:
                signal_col = signal_cols[0]
            else:
                raise ValueError(f"Signal column not found in data")

        # Convert to numeric but KEEP NaN values - NO CLEANSING
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df[signal_col] = pd.to_numeric(df[signal_col], errors='coerce')

        logger.info(f"  Total rows: {len(df):,}")
        logger.info(f"  Time NaN: {df[time_col].isna().sum():,}")
        logger.info(f"  Signal NaN: {df[signal_col].isna().sum():,}")

        # Get metadata
        case_info = self.util.get_case_info(case_id)

        # Prepare output
        output_data = {
            'case_id': case_id,
            'track_name': track_name,
            'track_id': track_id,
            'num_samples': len(df),
            'expected_sampling_rate': self.PPG_TRACKS.get(track_name),
            'time_column': time_col,
            'signal_column': signal_col,
            'case_info': case_info
        }

        # Save RAW data as CSV (with NaN values intact)
        safe_track_name = track_name.replace('/', '_')
        csv_filename = f"case_{case_id}_{safe_track_name}_raw.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Rename columns and save
        df_output = df[[time_col, signal_col]].copy()
        df_output.columns = ['time', 'ppg']
        df_output.to_csv(csv_path, index=False)

        output_data['csv_file'] = csv_path
        logger.info(f"Saved RAW PPG data to: {csv_path}")

        # Save metadata
        json_filename = f"case_{case_id}_{safe_track_name}_raw_metadata.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        output_data['metadata_file'] = json_path
        logger.info(f"Saved metadata to: {json_path}")
        logger.info(f"RAW extraction complete: {len(df):,} samples")

        return output_data

    def extract_ppg(self, case_id: int, track_name: str,
                    output_dir: str = './ppg_data') -> Dict:
        """
        Extract PPG signal for a specific case and track.

        Args:
            case_id: VitalDB case ID
            track_name: PPG track name (e.g., 'SNUADC/PLETH')
            output_dir: Directory to save extracted data

        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting PPG signal: Case {case_id}, Track: {track_name}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get case tracks
        tracks = self.util.get_case_tracks(case_id)

        # Find matching track
        matching_track = None
        for track in tracks:
            if track['tname'] == track_name:
                matching_track = track
                break

        if not matching_track:
            raise ValueError(f"Track '{track_name}' not found for case {case_id}")

        # Download track data
        track_id = matching_track['tid']
        logger.info(f"Downloading data for track ID: {track_id}")
        raw_data = self.util.download_track_data(track_id)

        if not raw_data:
            raise ValueError(f"No data available for track {track_name}")

        # Convert to DataFrame
        df = pd.DataFrame(raw_data)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Get time and signal columns
        time_col = 'Time'
        signal_col = track_name

        if time_col not in df.columns:
            # Try to find time column
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                time_col = time_cols[0]
            else:
                raise ValueError(f"Time column not found in data")

        if signal_col not in df.columns:
            # Try to find signal column
            signal_cols = [col for col in df.columns if col != time_col]
            if signal_cols:
                signal_col = signal_cols[0]
            else:
                raise ValueError(f"Signal column not found in data")

        # Convert to numeric
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df[signal_col] = pd.to_numeric(df[signal_col], errors='coerce')

        # Check for empty signal before dropping NaN
        total_rows = len(df)
        non_empty_signal = df[signal_col].notna().sum()

        logger.info(f"  Total rows: {total_rows:,}")
        logger.info(f"  Non-empty signal values: {non_empty_signal:,} ({non_empty_signal/total_rows*100:.1f}%)")

        if non_empty_signal == 0:
            raise ValueError(f"No valid signal data found in track {track_name}")

        if non_empty_signal < 100:
            raise ValueError(f"Insufficient data: only {non_empty_signal} valid samples found")

        # Remove NaN values from signal column only
        df = df.dropna(subset=[signal_col])

        # Check time column separately - if missing, create sequential time values
        if df[time_col].isna().all():
            logger.warning(f"  Time column has all NaN values, creating sequential time indices")
            # Assume expected sampling rate if available
            expected_rate = self.PPG_TRACKS.get(track_name)
            if expected_rate:
                df[time_col] = np.arange(len(df)) / expected_rate
            else:
                df[time_col] = np.arange(len(df))
        else:
            # Remove rows where time is NaN
            df = df.dropna(subset=[time_col])

        logger.info(f"  Valid samples after cleaning: {len(df):,}")

        # Calculate sampling rate
        time_diffs = df[time_col].diff().dropna()
        median_interval = time_diffs.median()
        estimated_sampling_rate = 1.0 / median_interval if median_interval > 0 else None

        # Get metadata
        case_info = self.util.get_case_info(case_id)

        # Prepare output
        output_data = {
            'case_id': case_id,
            'track_name': track_name,
            'track_id': track_id,
            'num_samples': len(df),
            'duration_seconds': df[time_col].max() - df[time_col].min(),
            'estimated_sampling_rate': estimated_sampling_rate,
            'expected_sampling_rate': self.PPG_TRACKS.get(track_name),
            'time_column': time_col,
            'signal_column': signal_col,
            'case_info': case_info
        }

        # Save data as CSV
        safe_track_name = track_name.replace('/', '_')
        csv_filename = f"case_{case_id}_{safe_track_name}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Rename columns for clarity
        df_output = df[[time_col, signal_col]].copy()
        df_output.columns = ['time', 'ppg']
        df_output.to_csv(csv_path, index=False)

        output_data['csv_file'] = csv_path
        logger.info(f"Saved PPG data to: {csv_path}")

        # Save metadata as JSON
        json_filename = f"case_{case_id}_{safe_track_name}_metadata.json"
        json_path = os.path.join(output_dir, json_filename)

        metadata = {k: v for k, v in output_data.items() if k != 'case_info'}
        metadata['case_info'] = case_info

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        output_data['metadata_file'] = json_path
        logger.info(f"Saved metadata to: {json_path}")

        # Log summary
        logger.info(f"Extraction complete:")
        logger.info(f"  Samples: {output_data['num_samples']:,}")
        logger.info(f"  Duration: {output_data['duration_seconds']:.2f} seconds")
        if estimated_sampling_rate:
            logger.info(f"  Sampling rate: {estimated_sampling_rate:.2f} Hz")

        return output_data

    def extract_all_ppg_tracks(self, case_id: int,
                               output_dir: str = './ppg_data') -> List[Dict]:
        """
        Extract all available PPG tracks for a case.

        Args:
            case_id: VitalDB case ID
            output_dir: Directory to save extracted data

        Returns:
            List of extraction results for each track
        """
        ppg_tracks = self.get_available_ppg_tracks(case_id)

        if not ppg_tracks:
            logger.warning(f"No PPG tracks found for case {case_id}")
            return []

        results = []
        for track in ppg_tracks:
            try:
                result = self.extract_ppg(case_id, track['tname'], output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {track['tname']}: {e}")
                continue

        return results

    def find_cases_with_ppg(self, track_name: Optional[str] = None) -> List[int]:
        """
        Find all cases that have PPG data.

        Args:
            track_name: Specific PPG track name (optional)

        Returns:
            List of case IDs with PPG data
        """
        logger.info("Finding cases with PPG data...")

        if track_name:
            # Find cases with specific track
            cases = self.util.get_cases_with_tracks([track_name])
        else:
            # Find cases with any PPG track
            ppg_track_names = list(self.PPG_TRACKS.keys())
            cases = self.util.get_cases_with_any_track(ppg_track_names)

        logger.info(f"Found {len(cases)} cases with PPG data")
        return cases

    def extract_best_ppg_track(self, case_id: int,
                               output_dir: str = './ppg_data',
                               preferred_tracks: Optional[List[str]] = None) -> Dict:
        """
        Extract the best available PPG track for a case.

        Tries multiple PPG tracks in order of preference until one succeeds.

        Args:
            case_id: VitalDB case ID
            output_dir: Output directory
            preferred_tracks: List of track names in order of preference

        Returns:
            Dictionary with extraction results
        """
        if preferred_tracks is None:
            # Default preference order (highest sampling rate first)
            preferred_tracks = [
                'SNUADC/PLETH',
                'Primus/PLETH',
                'Solar8000/PLETH',
                'BIS/PLETH'
            ]

        logger.info(f"Finding best PPG track for case {case_id}...")

        # Get available tracks
        available_tracks = self.get_available_ppg_tracks(case_id)

        if not available_tracks:
            raise ValueError(f"No PPG tracks available for case {case_id}")

        available_track_names = [t['tname'] for t in available_tracks]

        # Try tracks in preference order
        for track_name in preferred_tracks:
            if track_name not in available_track_names:
                continue

            logger.info(f"  Trying track: {track_name}")

            try:
                result = self.extract_ppg(case_id, track_name, output_dir)
                logger.info(f"  ✓ Successfully extracted {track_name}")
                return result
            except ValueError as e:
                logger.warning(f"  ✗ {track_name} failed: {e}")
                continue
            except Exception as e:
                logger.warning(f"  ✗ {track_name} error: {e}")
                continue

        # If none of the preferred tracks worked, try any remaining tracks
        for track in available_tracks:
            track_name = track['tname']

            if track_name in preferred_tracks:
                continue  # Already tried

            logger.info(f"  Trying alternate track: {track_name}")

            try:
                result = self.extract_ppg(case_id, track_name, output_dir)
                logger.info(f"  ✓ Successfully extracted {track_name}")
                return result
            except Exception as e:
                logger.warning(f"  ✗ {track_name} failed: {e}")
                continue

        raise ValueError(f"No valid PPG data found for case {case_id} in any available track")


def main():
    """Example usage of PPG extractor."""
    print("VitalDB PPG Extractor")
    print("=" * 60)

    extractor = PPGExtractor()

    # Example 1: Find cases with PPG
    print("\n[Example 1] Finding cases with SNUADC/PLETH...")
    cases = extractor.find_cases_with_ppg('SNUADC/PLETH')
    print(f"Found {len(cases)} cases")
    print(f"First 10 cases: {cases[:10]}")

    # Example 2: Check available PPG tracks for case 1
    print("\n[Example 2] Checking PPG tracks for case 1...")
    ppg_tracks = extractor.get_available_ppg_tracks(1)
    if ppg_tracks:
        print(f"Available PPG tracks:")
        for track in ppg_tracks:
            print(f"  - {track['tname']}")
    else:
        print("No PPG tracks found for case 1")

    # Example 3: Extract PPG data
    if ppg_tracks:
        print(f"\n[Example 3] Extracting PPG data...")
        track_name = ppg_tracks[0]['tname']
        result = extractor.extract_ppg(1, track_name, './ppg_data')

        print(f"\nExtraction complete!")
        print(f"  File: {result['csv_file']}")
        print(f"  Samples: {result['num_samples']:,}")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")
        if result['estimated_sampling_rate']:
            print(f"  Sampling rate: {result['estimated_sampling_rate']:.2f} Hz")


if __name__ == "__main__":
    main()
