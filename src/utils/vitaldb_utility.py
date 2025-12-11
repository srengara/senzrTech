"""
VitalDB Utility Class
=====================
A utility class for accessing and listing case IDs from the VitalDB dataset.

VitalDB is a surgical patient biosignal dataset containing 6,388 cases.
More information: https://vitaldb.net/dataset/

Usage:
    from src.utils.vitaldb_utility import VitalDBUtility

    # Initialize the utility
    util = VitalDBUtility()

    # Get all case IDs
    case_ids = util.get_all_case_ids()
    print(f"Total cases: {len(case_ids)}")

    # Get case IDs with specific tracks
    ecg_cases = util.get_cases_with_tracks(['ECG_II', 'ART'])
"""

import requests
import gzip
import io
import csv
from typing import List, Optional, Set
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VitalDBUtility:
    """Utility class for accessing VitalDB dataset case IDs."""

    # VitalDB API endpoints
    API_BASE_URL = "https://api.vitaldb.net"
    CASES_ENDPOINT = f"{API_BASE_URL}/cases"
    TRACKS_ENDPOINT = f"{API_BASE_URL}/trks"
    LABS_ENDPOINT = f"{API_BASE_URL}/labs"

    def __init__(self):
        """Initialize the VitalDB utility."""
        self._case_ids_cache = None
        self._tracks_cache = None

    def get_all_case_ids(self, use_cache: bool = True) -> List[int]:
        """
        Get all case IDs from the VitalDB dataset.

        This method fetches clinical information from the VitalDB API
        and extracts all unique case IDs.

        Args:
            use_cache: If True, use cached results if available

        Returns:
            List of case IDs (integers)

        Example:
            >>> util = VitalDBUtility()
            >>> case_ids = util.get_all_case_ids()
            >>> print(f"Total cases: {len(case_ids)}")
        """
        if use_cache and self._case_ids_cache is not None:
            logger.info(f"Returning {len(self._case_ids_cache)} cached case IDs")
            return self._case_ids_cache

        try:
            logger.info("Fetching case IDs from VitalDB API...")
            response = requests.get(self.CASES_ENDPOINT, timeout=30)
            response.raise_for_status()

            # Try to decompress if gzip, otherwise use plain text
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
            except (gzip.BadGzipFile, OSError):
                # Not gzipped, decode as plain text
                content = response.content.decode('utf-8-sig')  # utf-8-sig removes BOM

            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(content))
            case_ids = []

            for row in csv_reader:
                if 'caseid' in row:
                    try:
                        case_ids.append(int(row['caseid']))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid case ID: {row.get('caseid')}")

            # Remove duplicates and sort
            case_ids = sorted(list(set(case_ids)))
            self._case_ids_cache = case_ids

            logger.info(f"Successfully retrieved {len(case_ids)} case IDs")
            return case_ids

        except requests.RequestException as e:
            logger.error(f"Failed to fetch case IDs: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing case IDs: {e}")
            raise

    def get_case_ids_by_track(self, use_cache: bool = True) -> dict:
        """
        Get all case IDs organized by available tracks.

        Returns:
            Dictionary mapping track names to lists of case IDs

        Example:
            >>> util = VitalDBUtility()
            >>> cases_by_track = util.get_case_ids_by_track()
            >>> print(f"Cases with ECG_II: {len(cases_by_track.get('ECG_II', []))}")
        """
        if use_cache and self._tracks_cache is not None:
            logger.info("Returning cached tracks data")
            return self._tracks_cache

        try:
            logger.info("Fetching tracks from VitalDB API...")
            response = requests.get(self.TRACKS_ENDPOINT, timeout=30)
            response.raise_for_status()

            # Try to decompress if gzip, otherwise use plain text
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
            except (gzip.BadGzipFile, OSError):
                # Not gzipped, decode as plain text
                content = response.content.decode('utf-8-sig')  # utf-8-sig removes BOM

            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(content))
            tracks_dict = {}

            for row in csv_reader:
                caseid = row.get('caseid')
                tname = row.get('tname')

                if caseid and tname:
                    try:
                        caseid_int = int(caseid)
                        if tname not in tracks_dict:
                            tracks_dict[tname] = []
                        tracks_dict[tname].append(caseid_int)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid data: caseid={caseid}, tname={tname}")

            # Remove duplicates and sort
            for track_name in tracks_dict:
                tracks_dict[track_name] = sorted(list(set(tracks_dict[track_name])))

            self._tracks_cache = tracks_dict
            logger.info(f"Successfully retrieved {len(tracks_dict)} unique tracks")
            return tracks_dict

        except requests.RequestException as e:
            logger.error(f"Failed to fetch tracks: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing tracks: {e}")
            raise

    def get_cases_with_tracks(self, track_names: List[str]) -> List[int]:
        """
        Get case IDs that contain all specified tracks.

        This is useful for finding cases that have specific biosignals available.

        Args:
            track_names: List of track names (e.g., ['ECG_II', 'ART'])

        Returns:
            List of case IDs that contain all specified tracks

        Example:
            >>> util = VitalDBUtility()
            >>> ecg_cases = util.get_cases_with_tracks(['ECG_II', 'ART'])
            >>> print(f"Cases with both ECG_II and ART: {len(ecg_cases)}")
        """
        if not track_names:
            return self.get_all_case_ids()

        tracks_dict = self.get_case_ids_by_track()

        # Find intersection of all specified tracks
        case_sets = []
        for track_name in track_names:
            if track_name in tracks_dict:
                case_sets.append(set(tracks_dict[track_name]))
            else:
                logger.warning(f"Track '{track_name}' not found in dataset")
                return []

        if not case_sets:
            return []

        # Intersection of all sets
        result = set.intersection(*case_sets)
        result_list = sorted(list(result))

        logger.info(f"Found {len(result_list)} cases with tracks: {track_names}")
        return result_list

    def get_available_tracks(self) -> List[str]:
        """
        Get a list of all available track names in the dataset.

        Returns:
            Sorted list of unique track names

        Example:
            >>> util = VitalDBUtility()
            >>> tracks = util.get_available_tracks()
            >>> print(f"Available tracks: {tracks[:10]}")
        """
        tracks_dict = self.get_case_ids_by_track()
        return sorted(tracks_dict.keys())

    def get_case_count(self) -> int:
        """
        Get the total number of cases in the dataset.

        Returns:
            Total number of cases
        """
        return len(self.get_all_case_ids())

    def get_cases_with_any_track(self, track_names: List[str]) -> List[int]:
        """
        Get case IDs that contain at least one of the specified tracks.

        Args:
            track_names: List of track names (e.g., ['ECG_II', 'ART'])

        Returns:
            List of case IDs that contain at least one of the specified tracks

        Example:
            >>> util = VitalDBUtility()
            >>> cases = util.get_cases_with_any_track(['ECG_II', 'ART'])
            >>> print(f"Cases with ECG_II or ART: {len(cases)}")
        """
        if not track_names:
            return []

        tracks_dict = self.get_case_ids_by_track()

        # Find union of all specified tracks
        case_sets = []
        for track_name in track_names:
            if track_name in tracks_dict:
                case_sets.append(set(tracks_dict[track_name]))
            else:
                logger.warning(f"Track '{track_name}' not found in dataset")

        if not case_sets:
            return []

        # Union of all sets
        result = set.union(*case_sets)
        result_list = sorted(list(result))

        logger.info(f"Found {len(result_list)} cases with any of tracks: {track_names}")
        return result_list

    def clear_cache(self):
        """Clear all cached data."""
        self._case_ids_cache = None
        self._tracks_cache = None
        logger.info("Cache cleared")

    def get_case_info(self, case_id: int) -> dict:
        """
        Get clinical information for a specific case.

        Args:
            case_id: The case ID to retrieve information for

        Returns:
            Dictionary containing clinical information for the case

        Example:
            >>> util = VitalDBUtility()
            >>> case_info = util.get_case_info(1)
            >>> print(f"Age: {case_info.get('age')}, Sex: {case_info.get('sex')}")
        """
        try:
            logger.info(f"Fetching case info for case ID {case_id}...")
            response = requests.get(self.CASES_ENDPOINT, timeout=30)
            response.raise_for_status()

            # Try to decompress if gzip, otherwise use plain text
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
            except (gzip.BadGzipFile, OSError):
                content = response.content.decode('utf-8-sig')

            # Parse CSV and find the specific case
            csv_reader = csv.DictReader(io.StringIO(content))
            for row in csv_reader:
                if row.get('caseid') == str(case_id):
                    logger.info(f"Found case {case_id}")
                    return dict(row)

            logger.warning(f"Case ID {case_id} not found")
            return {}

        except Exception as e:
            logger.error(f"Error fetching case info: {e}")
            raise

    def get_case_tracks(self, case_id: int) -> List[dict]:
        """
        Get all available tracks for a specific case.

        Args:
            case_id: The case ID to get tracks for

        Returns:
            List of dictionaries containing track information (tname, tid)

        Example:
            >>> util = VitalDBUtility()
            >>> tracks = util.get_case_tracks(1)
            >>> for track in tracks[:5]:
            ...     print(f"{track['tname']}: {track['tid']}")
        """
        try:
            logger.info(f"Fetching tracks for case ID {case_id}...")
            response = requests.get(self.TRACKS_ENDPOINT, timeout=30)
            response.raise_for_status()

            # Try to decompress if gzip, otherwise use plain text
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
            except (gzip.BadGzipFile, OSError):
                content = response.content.decode('utf-8-sig')

            # Parse CSV and filter by case ID
            csv_reader = csv.DictReader(io.StringIO(content))
            case_tracks = []

            for row in csv_reader:
                if row.get('caseid') == str(case_id):
                    case_tracks.append({
                        'tname': row.get('tname'),
                        'tid': row.get('tid')
                    })

            logger.info(f"Found {len(case_tracks)} tracks for case {case_id}")
            return case_tracks

        except Exception as e:
            logger.error(f"Error fetching case tracks: {e}")
            raise

    def download_track_data(self, track_id: str) -> List[dict]:
        """
        Download actual biosignal data for a specific track.

        Args:
            track_id: The track ID (tid) to download data for

        Returns:
            List of dictionaries with 'time' and 'value' keys

        Example:
            >>> util = VitalDBUtility()
            >>> data = util.download_track_data('12345')
            >>> print(f"Downloaded {len(data)} data points")
        """
        try:
            logger.info(f"Downloading track data for tid={track_id}...")
            url = f"{self.API_BASE_URL}/{track_id}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Try to decompress if gzip, otherwise use plain text
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
            except (gzip.BadGzipFile, OSError):
                content = response.content.decode('utf-8-sig')

            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(content))
            data = []

            for row in csv_reader:
                data.append(dict(row))

            logger.info(f"Downloaded {len(data)} data points for track {track_id}")
            return data

        except Exception as e:
            logger.error(f"Error downloading track data: {e}")
            raise

    def load_track(self, case_id: int, track_name: str) -> Optional[dict]:
        """
        Load track data for a specific case and track name.

        Args:
            case_id: VitalDB case ID
            track_name: Name of the track (e.g., 'Laboratory/GLU', 'SNUADC/PLETH')

        Returns:
            Dictionary with 'time' and 'value' arrays, or None if not found

        Example:
            >>> util = VitalDBUtility()
            >>> data = util.load_track(1, 'Laboratory/GLU')
            >>> print(f"Time: {data['time'][:5]}")
            >>> print(f"Value: {data['value'][:5]}")
        """
        try:
            # First, get all tracks for this case
            case_tracks = self.get_case_tracks(case_id)

            # Find the matching track
            track_id = None
            for track in case_tracks:
                if track['tname'] == track_name:
                    track_id = track['tid']
                    break

            if track_id is None:
                logger.warning(f"Track '{track_name}' not found for case {case_id}")
                return None

            # Download the track data
            raw_data = self.download_track_data(track_id)

            if not raw_data:
                logger.warning(f"No data in track '{track_name}' for case {case_id}")
                return None

            # Convert to numpy-friendly format
            times = []
            values = []

            for row in raw_data:
                try:
                    time_val = float(row.get('time', 0))
                    data_val = float(row.get('val', row.get('value', 0)))
                    times.append(time_val)
                    values.append(data_val)
                except (ValueError, TypeError):
                    continue

            result = {
                'time': np.array(times),
                'value': np.array(values)
            }

            logger.info(f"Loaded {len(times)} data points from track '{track_name}'")
            return result

        except Exception as e:
            logger.error(f"Error loading track data: {e}")
            return None

    def download_case_data(self, case_id: int, track_names: Optional[List[str]] = None,
                          output_dir: str = '.') -> dict:
        """
        Download complete case data including clinical info and biosignals.

        Args:
            case_id: The case ID to download
            track_names: Optional list of specific track names to download.
                        If None, downloads all available tracks.
            output_dir: Directory to save downloaded data (default: current directory)

        Returns:
            Dictionary containing:
                - 'case_info': Clinical information
                - 'tracks': List of track metadata
                - 'data_files': List of saved file paths

        Example:
            >>> util = VitalDBUtility()
            >>> result = util.download_case_data(1, ['SNUADC/ECG_II', 'SNUADC/ART'])
            >>> print(f"Downloaded {len(result['data_files'])} tracks")
        """
        import os
        import json

        try:
            logger.info(f"Starting download for case {case_id}...")

            # Create output directory
            case_dir = os.path.join(output_dir, f"case_{case_id}")
            os.makedirs(case_dir, exist_ok=True)

            # Get case info
            case_info = self.get_case_info(case_id)
            if not case_info:
                raise ValueError(f"Case {case_id} not found")

            # Save case info
            info_file = os.path.join(case_dir, "case_info.json")
            with open(info_file, 'w') as f:
                json.dump(case_info, f, indent=2)
            logger.info(f"Saved case info to {info_file}")

            # Get all tracks for this case
            all_tracks = self.get_case_tracks(case_id)

            # Filter tracks if specific names requested
            if track_names:
                tracks_to_download = [t for t in all_tracks if t['tname'] in track_names]
                if not tracks_to_download:
                    logger.warning(f"No matching tracks found for {track_names}")
            else:
                tracks_to_download = all_tracks

            # Save track metadata
            tracks_file = os.path.join(case_dir, "tracks.json")
            with open(tracks_file, 'w') as f:
                json.dump(tracks_to_download, f, indent=2)
            logger.info(f"Saved track metadata to {tracks_file}")

            # Download track data
            data_files = []
            for i, track in enumerate(tracks_to_download, 1):
                track_name = track['tname']
                track_id = track['tid']

                logger.info(f"Downloading track {i}/{len(tracks_to_download)}: {track_name}")

                try:
                    track_data = self.download_track_data(track_id)

                    # Save track data as CSV
                    safe_name = track_name.replace('/', '_').replace('\\', '_')
                    data_file = os.path.join(case_dir, f"{safe_name}.csv")

                    with open(data_file, 'w', newline='') as f:
                        if track_data:
                            writer = csv.DictWriter(f, fieldnames=track_data[0].keys())
                            writer.writeheader()
                            writer.writerows(track_data)
                            data_files.append(data_file)
                            logger.info(f"Saved {len(track_data)} data points to {data_file}")
                        else:
                            logger.warning(f"No data for track {track_name}")

                except Exception as e:
                    logger.error(f"Failed to download track {track_name}: {e}")
                    continue

            result = {
                'case_id': case_id,
                'case_info': case_info,
                'tracks': tracks_to_download,
                'data_files': data_files,
                'output_dir': case_dir
            }

            logger.info(f"Download complete! Saved to {case_dir}")
            return result

        except Exception as e:
            logger.error(f"Error downloading case data: {e}")
            raise


# Convenience function for quick access
def get_all_case_ids() -> List[int]:
    """
    Convenience function to quickly get all case IDs.

    Returns:
        List of all case IDs from VitalDB dataset
    """
    util = VitalDBUtility()
    return util.get_all_case_ids()


if __name__ == "__main__":
    # Example usage
    print("VitalDB Utility - Example Usage")
    print("=" * 50)

    # Create utility instance
    util = VitalDBUtility()

    # Get all case IDs
    print("\n1. Getting all case IDs...")
    case_ids = util.get_all_case_ids()
    print(f"   Total cases: {len(case_ids)}")
    print(f"   First 10 case IDs: {case_ids[:10]}")
    print(f"   Last 10 case IDs: {case_ids[-10:]}")

    # Get available tracks
    print("\n2. Getting available tracks...")
    tracks = util.get_available_tracks()
    print(f"   Total unique tracks: {len(tracks)}")
    print(f"   First 20 tracks: {tracks[:20]}")

    # Get cases with specific tracks
    print("\n3. Finding cases with specific tracks...")
    example_tracks = ['SNUADC/ECG_II', 'SNUADC/ART']
    cases_with_tracks = util.get_cases_with_tracks(example_tracks)
    print(f"   Cases with {example_tracks}: {len(cases_with_tracks)}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")
