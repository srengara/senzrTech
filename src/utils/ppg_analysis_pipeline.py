"""
PPG Analysis Pipeline
=====================
Complete pipeline for PPG extraction, segmentation, and visualization.

Usage:
    # Single case
    python ppg_analysis_pipeline.py --case-id 1

    # Batch processing
    python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10
"""

import argparse
import os
import sys
import json
from typing import List, Optional, Dict
from src.data_extraction.ppg_extractor import PPGExtractor
from src.data_extraction.ppg_segmentation import PPGSegmenter
from src.data_extraction.ppg_plotter import PPGPlotter
from src.data_extraction.ppg_visualizer import PPGHTMLVisualizer


def select_ppg_tracks(case_ids: List[int], extractor: PPGExtractor) -> Optional[str]:
    """
    Interactive track selection based on available tracks across multiple cases.

    Args:
        case_ids: List of case IDs to check
        extractor: PPGExtractor instance

    Returns:
        Selected track name or None for auto-selection
    """
    print("\n" + "=" * 70)
    print("PPG TRACK SELECTION")
    print("=" * 70)

    # Analyze tracks across all cases
    print(f"\nAnalyzing PPG tracks across {len(case_ids)} case(s)...")

    track_availability = {}

    for case_id in case_ids[:min(10, len(case_ids))]:  # Check up to 10 cases for speed
        try:
            tracks = extractor.get_available_ppg_tracks(case_id)
            for track in tracks:
                track_name = track['tname']
                if track_name not in track_availability:
                    track_availability[track_name] = {
                        'count': 0,
                        'cases': [],
                        'expected_rate': extractor.PPG_TRACKS.get(track_name, 'Unknown')
                    }
                track_availability[track_name]['count'] += 1
                track_availability[track_name]['cases'].append(case_id)
        except Exception as e:
            print(f"  Warning: Could not check case {case_id}: {e}")
            continue

    if not track_availability:
        print("\n✗ No PPG tracks found in the specified cases")
        return None

    # Display available tracks
    print(f"\nAvailable PPG tracks (checked {min(10, len(case_ids))} cases):")
    print("-" * 70)

    sorted_tracks = sorted(track_availability.items(),
                          key=lambda x: x[1]['count'],
                          reverse=True)

    for i, (track_name, info) in enumerate(sorted_tracks, 1):
        availability_pct = (info['count'] / min(10, len(case_ids))) * 100
        print(f"{i}. {track_name}")
        print(f"   Available in: {info['count']}/{min(10, len(case_ids))} cases ({availability_pct:.0f}%)")
        print(f"   Expected rate: {info['expected_rate']} Hz")
        print(f"   Example cases: {info['cases'][:5]}")
        print()

    # User selection
    print("=" * 70)
    print("Select PPG track to use:")
    print()
    print("  0 - Auto-select best available track for each case (RECOMMENDED)")

    for i, (track_name, _) in enumerate(sorted_tracks, 1):
        print(f"  {i} - {track_name}")

    print()

    while True:
        try:
            choice = input("Enter your choice (0-{}): ".format(len(sorted_tracks)))
            choice = int(choice)

            if choice == 0:
                print("\n✓ Will auto-select best available track for each case")
                return None
            elif 1 <= choice <= len(sorted_tracks):
                selected_track = sorted_tracks[choice - 1][0]
                print(f"\n✓ Selected: {selected_track}")
                return selected_track
            else:
                print(f"Invalid choice. Please enter 0-{len(sorted_tracks)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            sys.exit(0)


def run_pipeline(case_id: int,
                track_name: Optional[str] = None,
                sampling_rate: float = None,
                output_dir: str = './ppg_analysis',
                skip_plots: bool = False):
    """
    Run complete PPG analysis pipeline.

    Args:
        case_id: VitalDB case ID
        track_name: PPG track name
        sampling_rate: Sampling rate in Hz (auto-detect if None)
        output_dir: Output directory
    """
    print("=" * 70)
    print("PPG ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Case ID: {case_id}")
    print(f"Track: {track_name}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract PPG data
    print("\n[STEP 1/5] Extracting PPG data from VitalDB...")
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

        # Try to extract the best available track
        if track_name and any(t['tname'] == track_name for t in available_tracks):
            # User specified a valid track
            print(f"\nAttempting to extract: {track_name}")
            try:
                result = extractor.extract_ppg(case_id, track_name, output_dir)
            except ValueError as e:
                print(f"WARNING: {track_name} failed: {e}")
                print("Trying alternative tracks...")
                result = extractor.extract_best_ppg_track(case_id, output_dir)
        else:
            # Use best available track
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
        return False

    # Step 2: Create initial plot
    print("\n[STEP 2/5] Creating initial signal plot...")
    print("-" * 70)

    try:
        plotter = PPGPlotter()
        plot_file = plotter.plot_ppg_signal(
            csv_file,
            end_time=30,
            title=f"PPG Signal - Case {case_id} - {track_name} (First 30s)",
            output_file=os.path.join(output_dir, f"case_{case_id}_initial_plot.png")
        )
        print(f"✓ Initial plot saved: {plot_file}")

    except Exception as e:
        print(f"ERROR in plotting: {e}")
        # Continue anyway

    # Step 3: Segment PPG signal
    print("\n[STEP 3/5] Segmenting PPG signal...")
    print("-" * 70)

    try:
        segmenter = PPGSegmenter(
            sampling_rate=sampling_rate,
            min_heart_rate=40.0,
            max_heart_rate=180.0
        )

        segmentation_file = os.path.join(output_dir, f"case_{case_id}_segmentation.json")
        seg_result = segmenter.segment_from_file(csv_file, segmentation_file)

        print(f"\n✓ Segmentation complete!")
        print(f"  Total pulses: {seg_result['summary']['total_pulses']}")
        print(f"  Valid pulses: {seg_result['summary']['valid_pulses']}")

        if seg_result['summary']['valid_pulses'] > 0:
            print(f"  Mean HR: {seg_result['summary']['mean_heart_rate']:.1f} ± "
                  f"{seg_result['summary']['std_heart_rate']:.1f} bpm")
            print(f"  Mean interval: {seg_result['summary']['mean_interval']*1000:.1f} ms")

    except Exception as e:
        print(f"ERROR in segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Create detailed plots
    if not skip_plots:
        print("\n[STEP 4/5] Creating segmentation plots...")
        print("-" * 70)

        try:
            # Plot segments
            segment_files = plotter.plot_ppg_segments(
                csv_file,
                start_time=0,
                duration=10,
                num_segments=5,
                output_dir=output_dir
            )
            print(f"✓ Created {len(segment_files)} segment plots")

        except Exception as e:
            print(f"WARNING: Could not create segment plots: {e}")
    else:
        print("\n[STEP 4/5] Skipping plot generation...")
        print("-" * 70)
        print("✓ Plots skipped for faster processing")

    # Step 5: Generate HTML report
    print("\n[STEP 5/5] Generating HTML report...")
    print("-" * 70)

    try:
        visualizer = PPGHTMLVisualizer()
        html_file = os.path.join(output_dir, f"case_{case_id}_report.html")

        visualizer.generate_html(csv_file, segmentation_file, html_file)

        print(f"\n✓ HTML report generated: {html_file}")

    except Exception as e:
        print(f"ERROR in HTML generation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files in: {output_dir}")
    print(f"  - PPG Data: {os.path.basename(csv_file)}")
    print(f"  - Segmentation: {os.path.basename(segmentation_file)}")
    print(f"  - HTML Report: {os.path.basename(html_file)}")
    print(f"\nOpen the HTML report in your browser:")
    print(f"  file://{os.path.abspath(html_file)}")
    print("=" * 70)

    return True


def run_batch_pipeline(start_case_id: int,
                      end_case_id: int,
                      track_name: Optional[str] = None,
                      sampling_rate: float = None,
                      output_base_dir: str = './ppg_batch_analysis',
                      interactive: bool = True,
                      skip_plots: bool = False):
    """
    Run pipeline for multiple cases.

    Args:
        start_case_id: Starting case ID
        end_case_id: Ending case ID (inclusive)
        track_name: PPG track name (None for auto-select)
        sampling_rate: Sampling rate in Hz
        output_base_dir: Base output directory
        interactive: Whether to prompt for track selection
        skip_plots: Skip plot generation for speed

    Returns:
        Dictionary with batch processing results
    """
    print("=" * 70)
    print("PPG BATCH ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Case range: {start_case_id} to {end_case_id}")
    print(f"Total cases: {end_case_id - start_case_id + 1}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 70)

    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Generate case ID list
    case_ids = list(range(start_case_id, end_case_id + 1))

    # Interactive track selection
    if interactive and track_name is None:
        extractor = PPGExtractor()
        track_name = select_ppg_tracks(case_ids, extractor)

    if track_name:
        print(f"\nUsing track: {track_name}")
    else:
        print(f"\nUsing auto-selection for each case")

    # Process each case
    results = {
        'successful': [],
        'failed': [],
        'total': len(case_ids),
        'track_used': track_name
    }

    print(f"\n" + "=" * 70)
    print("PROCESSING CASES")
    print("=" * 70)

    for i, case_id in enumerate(case_ids, 1):
        print(f"\n[{i}/{len(case_ids)}] Processing case {case_id}...")
        print("-" * 70)

        # Create case-specific output directory
        case_output_dir = os.path.join(output_base_dir, f"case_{case_id}")

        try:
            success = run_pipeline(
                case_id=case_id,
                track_name=track_name,
                sampling_rate=sampling_rate,
                output_dir=case_output_dir,
                skip_plots=skip_plots
            )

            if success:
                results['successful'].append({
                    'case_id': case_id,
                    'output_dir': case_output_dir
                })
                print(f"\n✓ Case {case_id} completed successfully")
            else:
                results['failed'].append({
                    'case_id': case_id,
                    'reason': 'Pipeline returned False'
                })
                print(f"\n✗ Case {case_id} failed")

        except Exception as e:
            results['failed'].append({
                'case_id': case_id,
                'reason': str(e)
            })
            print(f"\n✗ Case {case_id} failed: {e}")
            continue

    # Final summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nTotal cases: {results['total']}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Success rate: {len(results['successful'])/results['total']*100:.1f}%")

    if results['successful']:
        print(f"\n✓ Successful cases:")
        for case in results['successful']:
            print(f"  - Case {case['case_id']}: {case['output_dir']}")

    if results['failed']:
        print(f"\n✗ Failed cases:")
        for case in results['failed']:
            print(f"  - Case {case['case_id']}: {case['reason']}")

    # Save results to JSON
    results_file = os.path.join(output_base_dir, 'batch_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBatch results saved to: {results_file}")
    print(f"All outputs in: {output_base_dir}")

    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='PPG Analysis Pipeline for VitalDB - Single or Batch Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single case analysis
  python ppg_analysis_pipeline.py --case-id 1

  # Batch processing with interactive track selection
  python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10

  # Batch processing with specific track (non-interactive)
  python ppg_analysis_pipeline.py --start-case-id 1 --end-case-id 10 --track "SNUADC/PLETH" --non-interactive

  # Batch processing, skip plots for speed
  python ppg_analysis_pipeline.py --start-case-id 100 --end-case-id 150 --skip-plots

  # Single case with specific track
  python ppg_analysis_pipeline.py --case-id 1 --track "Solar8000/PLETH"

  # List available tracks for a case
  python ppg_analysis_pipeline.py --case-id 1 --list-tracks

  # Custom output directory
  python ppg_analysis_pipeline.py --case-id 1 --output ./my_analysis
        """
    )

    # Mutually exclusive group for single vs batch
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--case-id', type=int,
                           help='Single case ID to analyze')
    mode_group.add_argument('--start-case-id', type=int,
                           help='Starting case ID for batch processing')

    # Batch-specific arguments
    parser.add_argument('--end-case-id', type=int,
                       help='Ending case ID for batch processing (inclusive, required with --start-case-id)')

    # Common arguments
    parser.add_argument('--track', type=str, default=None,
                       help='PPG track name (default: auto-select best track)')
    parser.add_argument('--sampling-rate', type=float, default=None,
                       help='Sampling rate in Hz (auto-detect if not specified)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: ./ppg_analysis or ./ppg_batch_analysis)')
    parser.add_argument('--list-tracks', action='store_true',
                       help='List available PPG tracks and exit (only with --case-id)')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Non-interactive mode (no track selection prompt for batch)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation for faster processing')

    args = parser.parse_args()

    # Validate batch mode arguments
    if args.start_case_id is not None:
        if args.end_case_id is None:
            parser.error("--end-case-id is required when using --start-case-id")
        if args.start_case_id > args.end_case_id:
            parser.error("--start-case-id must be less than or equal to --end-case-id")
        if args.list_tracks:
            parser.error("--list-tracks can only be used with --case-id")

    # List tracks mode (single case only)
    if args.list_tracks:
        print(f"Checking PPG tracks for case {args.case_id}...")
        extractor = PPGExtractor()
        tracks = extractor.get_available_ppg_tracks(args.case_id)

        if tracks:
            print(f"\nAvailable PPG tracks for case {args.case_id}:")
            for track in tracks:
                expected_rate = PPGExtractor.PPG_TRACKS.get(track['tname'], 'Unknown')
                print(f"  - {track['tname']} (expected rate: {expected_rate} Hz)")
        else:
            print(f"No PPG tracks found for case {args.case_id}")
        return

    # Single case mode
    if args.case_id is not None:
        output_dir = args.output if args.output else './ppg_analysis'

        success = run_pipeline(
            case_id=args.case_id,
            track_name=args.track,
            sampling_rate=args.sampling_rate,
            output_dir=output_dir,
            skip_plots=args.skip_plots
        )

        sys.exit(0 if success else 1)

    # Batch mode
    else:
        output_dir = args.output if args.output else './ppg_batch_analysis'

        results = run_batch_pipeline(
            start_case_id=args.start_case_id,
            end_case_id=args.end_case_id,
            track_name=args.track,
            sampling_rate=args.sampling_rate,
            output_base_dir=output_dir,
            interactive=not args.non_interactive,
            skip_plots=args.skip_plots
        )

        # Exit code based on success rate
        success_rate = len(results['successful']) / results['total']
        sys.exit(0 if success_rate > 0.5 else 1)


if __name__ == "__main__":
    main()
