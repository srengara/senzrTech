"""
PPG Signal Plotter
==================
Visualization utilities for PPG signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, Tuple, List
import os

matplotlib.use('Agg')  # Non-interactive backend for server use


class PPGPlotter:
    """Plotter for PPG signals."""

    def __init__(self, figsize: Tuple[int, int] = (15, 6)):
        """
        Initialize plotter.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize

    def load_ppg_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load PPG data from CSV file.

        Args:
            csv_file: Path to CSV file with time and ppg columns

        Returns:
            DataFrame with PPG data
        """
        df = pd.read_csv(csv_file)

        # Ensure required columns exist
        if 'time' not in df.columns or 'ppg' not in df.columns:
            raise ValueError("CSV must contain 'time' and 'ppg' columns")

        return df

    def plot_ppg_signal(self, csv_file: str,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       title: Optional[str] = None,
                       output_file: Optional[str] = None,
                       show_plot: bool = False) -> str:
        """
        Plot PPG signal.

        Args:
            csv_file: Path to CSV file with PPG data
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end)
            title: Plot title
            output_file: Output file path (if None, auto-generate)
            show_plot: Whether to show the plot

        Returns:
            Path to saved plot
        """
        # Load data
        df = self.load_ppg_data(csv_file)

        # Filter time range
        if start_time is not None:
            df = df[df['time'] >= start_time]
        if end_time is not None:
            df = df[df['time'] <= end_time]

        if len(df) == 0:
            raise ValueError("No data in specified time range")

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(df['time'], df['ppg'], 'b-', linewidth=0.8, alpha=0.8)

        # Labels and title
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('PPG Amplitude', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('PPG Signal', fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Mean: {df['ppg'].mean():.2f}\n"
        stats_text += f"Std: {df['ppg'].std():.2f}\n"
        stats_text += f"Min: {df['ppg'].min():.2f}\n"
        stats_text += f"Max: {df['ppg'].max():.2f}\n"
        stats_text += f"Samples: {len(df):,}"

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9,
                family='monospace')

        plt.tight_layout()

        # Save plot
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_file = f"{base_name}_plot.png"

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_file

    def plot_ppg_segments(self, csv_file: str,
                         start_time: Optional[float] = None,
                         duration: float = 10.0,
                         num_segments: int = 3,
                         output_dir: Optional[str] = None) -> List[str]:
        """
        Plot multiple segments of PPG signal.

        Args:
            csv_file: Path to CSV file
            start_time: Start time for first segment
            duration: Duration of each segment in seconds
            num_segments: Number of segments to plot
            output_dir: Output directory for plots

        Returns:
            List of output file paths
        """
        df = self.load_ppg_data(csv_file)

        if start_time is None:
            start_time = df['time'].min()

        if output_dir is None:
            output_dir = os.path.dirname(csv_file) or '.'

        output_files = []

        for i in range(num_segments):
            segment_start = start_time + (i * duration)
            segment_end = segment_start + duration

            if segment_end > df['time'].max():
                break

            output_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(csv_file))[0]}_segment_{i+1}.png"
            )

            title = f"PPG Signal Segment {i+1} ({segment_start:.1f}s - {segment_end:.1f}s)"

            try:
                self.plot_ppg_signal(
                    csv_file,
                    start_time=segment_start,
                    end_time=segment_end,
                    title=title,
                    output_file=output_file
                )
                output_files.append(output_file)
            except Exception as e:
                print(f"Failed to plot segment {i+1}: {e}")
                continue

        return output_files

    def plot_ppg_with_overlay(self, csv_file: str,
                             overlay_data: pd.DataFrame,
                             overlay_label: str = 'Overlay',
                             output_file: Optional[str] = None,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None) -> str:
        """
        Plot PPG signal with overlay data.

        Args:
            csv_file: Path to CSV file with original PPG
            overlay_data: DataFrame with time and value columns for overlay
            overlay_label: Label for overlay data
            output_file: Output file path
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Path to saved plot
        """
        # Load original PPG
        df = self.load_ppg_data(csv_file)

        # Filter time range
        if start_time is not None:
            df = df[df['time'] >= start_time]
            overlay_data = overlay_data[overlay_data['time'] >= start_time]
        if end_time is not None:
            df = df[df['time'] <= end_time]
            overlay_data = overlay_data[overlay_data['time'] <= end_time]

        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=self.figsize)

        # Plot original PPG
        color = 'tab:blue'
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('PPG Amplitude', color=color, fontsize=12)
        ax1.plot(df['time'], df['ppg'], color=color, linewidth=0.8, alpha=0.7, label='PPG')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Plot overlay
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(overlay_label, color=color, fontsize=12)

        if 'value' in overlay_data.columns:
            ax2.plot(overlay_data['time'], overlay_data['value'],
                    color=color, linewidth=1.5, alpha=0.8, label=overlay_label)
        elif len(overlay_data.columns) > 1:
            # Use second column
            col = overlay_data.columns[1]
            ax2.plot(overlay_data['time'], overlay_data[col],
                    color=color, linewidth=1.5, alpha=0.8, label=overlay_label)

        ax2.tick_params(axis='y', labelcolor=color)

        # Title
        plt.title('PPG Signal with Overlay', fontsize=14, fontweight='bold')

        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()

        # Save
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_file = f"{base_name}_overlay.png"

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Overlay plot saved to: {output_file}")
        return output_file

    def plot_ppg_comparison(self, csv_files: List[str],
                           labels: Optional[List[str]] = None,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           output_file: str = 'ppg_comparison.png') -> str:
        """
        Plot multiple PPG signals for comparison.

        Args:
            csv_files: List of CSV file paths
            labels: Labels for each signal
            start_time: Start time in seconds
            end_time: End time in seconds
            output_file: Output file path

        Returns:
            Path to saved plot
        """
        if labels is None:
            labels = [f"Signal {i+1}" for i in range(len(csv_files))]

        fig, axes = plt.subplots(len(csv_files), 1,
                                figsize=(self.figsize[0], self.figsize[1] * len(csv_files)),
                                sharex=True)

        if len(csv_files) == 1:
            axes = [axes]

        for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
            df = self.load_ppg_data(csv_file)

            # Filter time range
            if start_time is not None:
                df = df[df['time'] >= start_time]
            if end_time is not None:
                df = df[df['time'] <= end_time]

            axes[i].plot(df['time'], df['ppg'], 'b-', linewidth=0.8, alpha=0.8)
            axes[i].set_ylabel('Amplitude', fontsize=11)
            axes[i].set_title(label, fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (seconds)', fontsize=12)

        plt.suptitle('PPG Signal Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Comparison plot saved to: {output_file}")
        return output_file


def main():
    """Example usage of PPG plotter."""
    print("PPG Plotter Examples")
    print("=" * 60)

    # Check if sample data exists
    sample_files = [f for f in os.listdir('.') if f.startswith('case_') and f.endswith('.csv')]

    if not sample_files:
        print("No PPG data files found. Please run ppg_extractor.py first.")
        return

    print(f"\nFound {len(sample_files)} PPG data file(s)")

    plotter = PPGPlotter()

    # Example 1: Plot first file
    print(f"\n[Example 1] Plotting {sample_files[0]}...")
    plot_file = plotter.plot_ppg_signal(
        sample_files[0],
        end_time=30,  # First 30 seconds
        title=f"PPG Signal - First 30 seconds"
    )

    # Example 2: Plot segments
    print(f"\n[Example 2] Plotting segments...")
    segment_files = plotter.plot_ppg_segments(
        sample_files[0],
        start_time=0,
        duration=10,
        num_segments=3
    )
    print(f"Created {len(segment_files)} segment plots")


if __name__ == "__main__":
    main()
