#!/usr/bin/env python
"""
VitalDB Raw Data Extractor
===========================
Simple standalone script to download raw PPG data from VitalDB.

Usage:
    python extract_vitaldb_data.py --case_id 1 --track SNUADC/PLETH
    python extract_vitaldb_data.py --start_case_id 1 --end_case_id 5 --track SNUADC/PLETH
    python extract_vitaldb_data.py --case_id 1 --track SNUADC/PLETH --output ./my_data
    python extract_vitaldb_data.py --case_id 1 --list-tracks

Features:
    - Downloads raw PPG data from VitalDB (no preprocessing)
    - Supports single case or range of cases
    - Lists available PPG tracks for a case
    - Saves data as CSV with metadata JSON
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_extraction.ppg_extractor import PPGExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_available_tracks(case_id):
    """
    List all available PPG tracks for a case.

    Args:
        case_id: VitalDB case ID
    """
    logger.info("=" * 70)
    logger.info(f"Available PPG Tracks for Case {case_id}")
    logger.info("=" * 70)

    extractor = PPGExtractor()

    try:
        tracks = extractor.get_available_ppg_tracks(case_id)

        if not tracks:
            logger.warning(f"No PPG tracks found for case {case_id}")
            return

        logger.info(f"\nFound {len(tracks)} PPG track(s):")
        for i, track in enumerate(tracks, 1):
            track_name = track['tname']
            expected_rate = extractor.PPG_TRACKS.get(track_name, 'Unknown')
            logger.info(f"  {i}. {track_name}")
            logger.info(f"     Expected sampling rate: {expected_rate} Hz")

        logger.info("\nTo extract a track, run:")
        logger.info(f"  python extract_vitaldb_data.py --case_id {case_id} --track {tracks[0]['tname']}")

    except Exception as e:
        logger.error(f"Failed to list tracks: {e}")
        raise


def extract_raw_data(case_id, track_name, output_dir):
    """
    Extract raw PPG data from VitalDB for a single case.

    Args:
        case_id: VitalDB case ID
        track_name: PPG track name (e.g., 'SNUADC/PLETH')
        output_dir: Output directory for files

    Returns:
        Dictionary with extraction results
    """
    logger.info("=" * 70)
    logger.info("VitalDB Raw Data Extractor")
    logger.info("=" * 70)
    logger.info(f"Case ID: {case_id}")
    logger.info(f"Track: {track_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize extractor
    extractor = PPGExtractor()

    # Extract raw data
    logger.info("Downloading raw data from VitalDB...")
    try:
        result = extractor.extract_ppg_raw(case_id, track_name, output_dir)

        logger.info("\n" + "=" * 70)
        logger.info("Extraction Complete!")
        logger.info("=" * 70)
        logger.info(f"✓ Total samples: {result['num_samples']:,}")
        logger.info(f"✓ Expected sampling rate: {result['expected_sampling_rate']} Hz")
        logger.info(f"✓ CSV file: {result['csv_file']}")
        logger.info(f"✓ Metadata file: {result['metadata_file']}")
        logger.info("=" * 70)

        return result

    except ValueError as e:
        logger.error(f"\n✗ Extraction failed: {e}")
        logger.info("\nTip: Use --list-tracks to see available tracks for this case")
        raise
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}")
        raise


def extract_multiple_cases(start_case_id, end_case_id, track_name, output_dir):
    """
    Extract raw PPG data from VitalDB for a range of cases.

    Args:
        start_case_id: Starting case ID (inclusive)
        end_case_id: Ending case ID (inclusive)
        track_name: PPG track name (e.g., 'SNUADC/PLETH')
        output_dir: Output directory for files

    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 70)
    logger.info("VitalDB Batch Raw Data Extractor")
    logger.info("=" * 70)
    logger.info(f"Case range: {start_case_id} to {end_case_id}")
    logger.info(f"Total cases: {end_case_id - start_case_id + 1}")
    logger.info(f"Track: {track_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    logger.info("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize extractor
    extractor = PPGExtractor()

    # Track statistics
    successful = []
    failed = []
    results = []

    # Process each case
    for case_id in range(start_case_id, end_case_id + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Case {case_id} ({case_id - start_case_id + 1}/{end_case_id - start_case_id + 1})")
        logger.info(f"{'='*70}")

        try:
            result = extractor.extract_ppg_raw(case_id, track_name, output_dir)

            logger.info(f"✓ Case {case_id}: SUCCESS")
            logger.info(f"  Samples: {result['num_samples']:,}")
            logger.info(f"  File: {result['csv_file']}")

            successful.append(case_id)
            results.append(result)

        except ValueError as e:
            logger.warning(f"✗ Case {case_id}: FAILED - {e}")
            failed.append({'case_id': case_id, 'error': str(e)})

        except Exception as e:
            logger.error(f"✗ Case {case_id}: ERROR - {e}")
            failed.append({'case_id': case_id, 'error': str(e)})

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH EXTRACTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total cases attempted: {end_case_id - start_case_id + 1}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Success rate: {len(successful)/(end_case_id - start_case_id + 1)*100:.1f}%")

    if successful:
        logger.info(f"\nSuccessful cases: {successful}")
        total_samples = sum(r['num_samples'] for r in results)
        logger.info(f"Total samples downloaded: {total_samples:,}")

    if failed:
        logger.info(f"\nFailed cases:")
        for fail in failed:
            logger.info(f"  Case {fail['case_id']}: {fail['error']}")

    logger.info("=" * 70)

    return {
        'successful': successful,
        'failed': failed,
        'results': results,
        'total_cases': end_case_id - start_case_id + 1,
        'success_rate': len(successful)/(end_case_id - start_case_id + 1)*100
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract raw PPG data from VitalDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available PPG tracks for a case
  python extract_vitaldb_data.py --case_id 1 --list-tracks

  # Extract a single case
  python extract_vitaldb_data.py --case_id 1 --track SNUADC/PLETH

  # Extract a range of cases (cases 1 through 5)
  python extract_vitaldb_data.py --start_case_id 1 --end_case_id 5 --track SNUADC/PLETH

  # Extract to custom directory
  python extract_vitaldb_data.py --case_id 1 --track SNUADC/PLETH --output ./my_data

  # Batch extraction with custom output
  python extract_vitaldb_data.py --start_case_id 1 --end_case_id 10 --track SNUADC/PLETH --output ./batch_data

Common PPG tracks:
  - SNUADC/PLETH      (500 Hz)
  - Primus/PLETH      (100 Hz)
  - Solar8000/PLETH   (62.5 Hz)
  - BIS/PLETH         (Variable)
        """
    )

    # Case selection (mutually exclusive groups)
    case_group = parser.add_mutually_exclusive_group(required=True)
    case_group.add_argument('--case_id', type=int,
                            help='Single VitalDB case ID (e.g., 1, 2, 3, ...)')
    case_group.add_argument('--start_case_id', type=int,
                            help='Starting case ID for batch extraction (use with --end_case_id)')

    # Range extraction
    parser.add_argument('--end_case_id', type=int,
                        help='Ending case ID for batch extraction (inclusive, use with --start_case_id)')

    # Optional arguments
    parser.add_argument('--track', type=str,
                        help='PPG track name (e.g., SNUADC/PLETH)')
    parser.add_argument('--output', type=str, default='./vitaldb_raw_data',
                        help='Output directory (default: ./vitaldb_raw_data)')
    parser.add_argument('--list-tracks', action='store_true',
                        help='List available PPG tracks for the case (only works with --case_id)')

    args = parser.parse_args()

    try:
        # Validate range arguments
        if args.start_case_id is not None:
            if args.end_case_id is None:
                logger.error("Error: --end_case_id is required when using --start_case_id")
                return 1
            if args.start_case_id > args.end_case_id:
                logger.error("Error: --start_case_id must be <= --end_case_id")
                return 1
            if args.list_tracks:
                logger.error("Error: --list-tracks only works with --case_id (not with batch mode)")
                return 1

        # List tracks mode
        if args.list_tracks:
            list_available_tracks(args.case_id)
            return 0

        # Validate track is provided
        if not args.track:
            logger.error("Error: --track is required (or use --list-tracks to see available tracks)")
            parser.print_help()
            return 1

        # Batch extraction mode
        if args.start_case_id is not None:
            summary = extract_multiple_cases(
                args.start_case_id,
                args.end_case_id,
                args.track,
                args.output
            )

            logger.info("\n✅ BATCH EXTRACTION COMPLETE!")
            logger.info(f"\nSuccessfully extracted {len(summary['successful'])} out of {summary['total_cases']} cases")
            logger.info(f"All files saved to: {args.output}")

            return 0 if summary['successful'] else 1

        # Single case extraction mode
        else:
            result = extract_raw_data(args.case_id, args.track, args.output)

            logger.info("\n✅ SUCCESS! Raw data has been downloaded.")
            logger.info(f"\nData file: {result['csv_file']}")
            logger.info(f"The CSV contains {result['num_samples']:,} raw samples with columns: time, ppg")

            return 0

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
