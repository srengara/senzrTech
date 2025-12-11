"""
PPG HTML Visualizer
===================
Generates interactive HTML reports with PPG signal and segmentation results.
"""

import pandas as pd
import numpy as np
import json
import os
import base64
from typing import Optional, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class PPGHTMLVisualizer:
    """Generate HTML visualization for PPG signals and segmentation."""

    def __init__(self):
        """Initialize visualizer."""
        pass

    def create_signal_plot(self, csv_file: str,
                          segmentation_data: Optional[Dict] = None,
                          max_duration: float = 30.0) -> str:
        """
        Create plot of PPG signal with segmentation overlay.

        Args:
            csv_file: Path to PPG CSV file
            segmentation_data: Segmentation results dictionary
            max_duration: Maximum duration to plot (seconds)

        Returns:
            Base64 encoded image string
        """
        # Load data
        df = pd.read_csv(csv_file)
        time = df['time'].values
        signal = df['ppg'].values

        # Limit duration
        mask = time <= (time[0] + max_duration)
        time = time[mask]
        signal = signal[mask]

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot signal
        ax.plot(time, signal, 'b-', linewidth=0.8, alpha=0.7, label='PPG Signal')

        # Overlay segmentation if available
        if segmentation_data and 'pulses' in segmentation_data:
            pulses = segmentation_data['pulses']

            # Plot peaks and onsets
            for pulse in pulses:
                if pulse['onset_time'] > time[-1]:
                    break

                # Onset marker
                if pulse['onset_time'] >= time[0]:
                    ax.axvline(pulse['onset_time'], color='g', alpha=0.3, linestyle='--', linewidth=0.8)

                # Peak marker
                if pulse['peak_time'] <= time[-1]:
                    ax.plot(pulse['peak_time'], signal[np.argmin(np.abs(time - pulse['peak_time']))],
                           'r*', markersize=8, alpha=0.7)

                # Color code by quality
                if pulse['valid']:
                    color = 'green'
                else:
                    color = 'orange'

                # Add quality annotation (every 5th pulse)
                if pulse['pulse_number'] % 5 == 0 and pulse['peak_time'] <= time[-1]:
                    ax.text(pulse['peak_time'], signal[np.argmin(np.abs(time - pulse['peak_time']))],
                           f"{pulse['quality']:.2f}",
                           fontsize=7, ha='center', va='bottom', color=color)

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('PPG Amplitude', fontsize=12)
        ax.set_title('PPG Signal with Segmentation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()

        # Convert to base64
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return img_base64

    def create_heart_rate_plot(self, segmentation_data: Dict) -> str:
        """
        Create heart rate variability plot.

        Args:
            segmentation_data: Segmentation results

        Returns:
            Base64 encoded image string
        """
        if 'pulses' not in segmentation_data or len(segmentation_data['pulses']) == 0:
            return ""

        pulses = segmentation_data['pulses']

        # Extract data
        times = [p['onset_time'] for p in pulses]
        heart_rates = [p['heart_rate'] for p in pulses]
        valid = [p['valid'] for p in pulses]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        # Plot 1: Heart rate over time
        valid_times = [t for t, v in zip(times, valid) if v]
        valid_hrs = [hr for hr, v in zip(heart_rates, valid) if v]
        invalid_times = [t for t, v in zip(times, valid) if not v]
        invalid_hrs = [hr for hr, v in zip(heart_rates, valid) if not v]

        if valid_times:
            ax1.plot(valid_times, valid_hrs, 'go-', linewidth=1.5, markersize=4,
                    alpha=0.7, label='Valid pulses')
        if invalid_times:
            ax1.plot(invalid_times, invalid_hrs, 'rx', markersize=6,
                    alpha=0.5, label='Invalid pulses')

        ax1.set_ylabel('Heart Rate (bpm)', fontsize=12)
        ax1.set_title('Heart Rate Over Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Inter-beat intervals
        intervals = [p['pulse_interval'] * 1000 for p in pulses]  # Convert to ms
        valid_intervals = [i for i, v in zip(intervals, valid) if v]
        invalid_intervals = [i for i, v in zip(intervals, valid) if not v]

        if valid_times:
            ax2.plot(valid_times, valid_intervals, 'bo-', linewidth=1.5, markersize=4,
                    alpha=0.7, label='Valid pulses')
        if invalid_times:
            ax2.plot(invalid_times, invalid_intervals, 'rx', markersize=6,
                    alpha=0.5, label='Invalid pulses')

        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Interval (ms)', fontsize=12)
        ax2.set_title('Pulse Intervals (RR Intervals)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Convert to base64
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return img_base64

    def generate_html(self, csv_file: str,
                     segmentation_file: str,
                     output_file: str = 'ppg_report.html') -> str:
        """
        Generate complete HTML report.

        Args:
            csv_file: Path to PPG CSV file
            segmentation_file: Path to segmentation JSON file
            output_file: Output HTML file path

        Returns:
            Path to generated HTML file
        """
        print(f"Generating HTML report...")

        # Load segmentation data
        with open(segmentation_file, 'r') as f:
            seg_data = json.load(f)

        # Load original data for metadata
        df = pd.read_csv(csv_file)

        # Generate plots
        print("  Creating signal plot...")
        signal_plot = self.create_signal_plot(csv_file, seg_data, max_duration=30)

        print("  Creating heart rate plot...")
        hr_plot = self.create_heart_rate_plot(seg_data)

        # Generate HTML
        html = self._generate_html_content(
            csv_file,
            seg_data,
            signal_plot,
            hr_plot
        )

        # Save HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"HTML report saved to: {output_file}")
        return output_file

    def _generate_html_content(self, csv_file: str,
                               seg_data: Dict,
                               signal_plot: str,
                               hr_plot: str) -> str:
        """Generate HTML content."""

        summary = seg_data.get('summary', {})
        params = seg_data.get('parameters', {})

        # Extract case info if available
        case_id = "Unknown"
        track_name = "Unknown"
        if 'input_file' in seg_data:
            filename = os.path.basename(seg_data['input_file'])
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == 'case':
                case_id = parts[1]
            if len(parts) >= 3:
                track_name = '_'.join(parts[2:]).replace('.csv', '')

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PPG Analysis Report - Case {case_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}

        .content {{
            padding: 30px;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .info-card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .info-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}

        .info-card p {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}

        .info-card .unit {{
            font-size: 0.6em;
            color: #666;
            font-weight: normal;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .plot {{
            text-align: center;
            margin: 20px 0;
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        .plot img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .valid-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .valid-badge.valid {{
            background: #d4edda;
            color: #155724;
        }}

        .valid-badge.invalid {{
            background: #f8d7da;
            color: #721c24;
        }}

        .quality-bar {{
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}

        .quality-fill {{
            height: 100%;
            background: linear-gradient(90deg, #dc3545, #ffc107, #28a745);
            transition: width 0.3s ease;
        }}

        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}

        .parameters {{
            background: #f1f3f5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}

        .parameters h3 {{
            color: #495057;
            margin-bottom: 15px;
        }}

        .parameters dl {{
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 10px;
        }}

        .parameters dt {{
            font-weight: 600;
            color: #495057;
        }}

        .parameters dd {{
            color: #6c757d;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🫀 PPG Analysis Report</h1>
            <p>Precision Interval Segmentation</p>
        </header>

        <div class="content">
            <!-- Summary Statistics -->
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Case ID</h3>
                        <p>{case_id}</p>
                    </div>
                    <div class="info-card">
                        <h3>Track Name</h3>
                        <p style="font-size: 1.2em;">{track_name}</p>
                    </div>
                    <div class="info-card">
                        <h3>Total Pulses</h3>
                        <p>{summary.get('total_pulses', 0)}</p>
                    </div>
                    <div class="info-card">
                        <h3>Valid Pulses</h3>
                        <p>{summary.get('valid_pulses', 0)}
                           <span class="unit">({summary.get('validity_rate', 0)*100:.1f}%)</span>
                        </p>
                    </div>
                    <div class="info-card">
                        <h3>Mean Heart Rate</h3>
                        <p>{summary.get('mean_heart_rate', 0):.1f}
                           <span class="unit">bpm</span>
                        </p>
                    </div>
                    <div class="info-card">
                        <h3>HR Std Dev</h3>
                        <p>{summary.get('std_heart_rate', 0):.1f}
                           <span class="unit">bpm</span>
                        </p>
                    </div>
                    <div class="info-card">
                        <h3>Mean Interval</h3>
                        <p>{summary.get('mean_interval', 0)*1000:.0f}
                           <span class="unit">ms</span>
                        </p>
                    </div>
                    <div class="info-card">
                        <h3>Signal Duration</h3>
                        <p>{summary.get('signal_duration', 0):.1f}
                           <span class="unit">seconds</span>
                        </p>
                    </div>
                </div>
            </div>

            <!-- Signal Plot -->
            <div class="section">
                <h2>📈 PPG Signal with Segmentation</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{signal_plot}" alt="PPG Signal">
                    <p style="margin-top: 15px; color: #666; font-size: 0.9em;">
                        <span style="color: red;">★</span> = Peak (systolic point) |
                        <span style="color: green;">--</span> = Onset (diastolic point) |
                        Numbers = Quality scores
                    </p>
                </div>
            </div>

            <!-- Heart Rate Plot -->
            {"" if not hr_plot else f'''
            <div class="section">
                <h2>💓 Heart Rate Variability</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{hr_plot}" alt="Heart Rate Variability">
                </div>
            </div>
            '''}

            <!-- Pulse Details Table -->
            <div class="section">
                <h2>📋 Pulse Details (First 50)</h2>
                {self._generate_pulse_table(seg_data.get('pulses', [])[:50])}
            </div>

            <!-- Parameters -->
            <div class="section">
                <h2>⚙️ Segmentation Parameters</h2>
                <div class="parameters">
                    <dl>
                        <dt>Sampling Rate:</dt>
                        <dd>{params.get('sampling_rate', 'N/A')} Hz</dd>

                        <dt>Min Heart Rate:</dt>
                        <dd>{params.get('min_heart_rate', 'N/A')} bpm</dd>

                        <dt>Max Heart Rate:</dt>
                        <dd>{params.get('max_heart_rate', 'N/A')} bpm</dd>

                        <dt>Quality Threshold:</dt>
                        <dd>{params.get('quality_threshold', 'N/A')}</dd>

                        <dt>Input File:</dt>
                        <dd>{os.path.basename(csv_file)}</dd>
                    </dl>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated by VitalDB PPG Analysis Toolkit</p>
            <p style="margin-top: 5px; font-size: 0.85em;">Precision Interval Segmentation Algorithm</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _generate_pulse_table(self, pulses: list) -> str:
        """Generate HTML table for pulse data."""
        if not pulses:
            return "<p>No pulse data available.</p>"

        rows = []
        for pulse in pulses:
            valid_badge = '<span class="valid-badge valid">✓ Valid</span>' if pulse['valid'] else '<span class="valid-badge invalid">✗ Invalid</span>'

            quality_percent = pulse['quality'] * 100

            row = f"""
            <tr>
                <td>{pulse['pulse_number']}</td>
                <td>{pulse['onset_time']:.2f}</td>
                <td>{pulse['peak_time']:.2f}</td>
                <td>{pulse['pulse_interval']*1000:.1f}</td>
                <td>{pulse['heart_rate']:.1f}</td>
                <td>{pulse['amplitude']:.2f}</td>
                <td>
                    <div class="quality-bar">
                        <div class="quality-fill" style="width: {quality_percent}%"></div>
                    </div>
                    {pulse['quality']:.3f}
                </td>
                <td>{valid_badge}</td>
            </tr>
            """
            rows.append(row)

        table = f"""
        <table>
            <thead>
                <tr>
                    <th>Pulse #</th>
                    <th>Onset Time (s)</th>
                    <th>Peak Time (s)</th>
                    <th>Interval (ms)</th>
                    <th>HR (bpm)</th>
                    <th>Amplitude</th>
                    <th>Quality</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

        return table


def main():
    """Example usage."""
    import os

    print("PPG HTML Visualizer Example")
    print("=" * 60)

    # Find segmentation files
    seg_files = [f for f in os.listdir('.') if f.endswith('_segmentation.json')]

    if not seg_files:
        print("No segmentation files found. Please run ppg_segmentation.py first.")
        return

    seg_file = seg_files[0]
    csv_file = seg_file.replace('_segmentation.json', '.csv')

    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    print(f"Creating HTML report for: {seg_file}")

    visualizer = PPGHTMLVisualizer()
    output_file = seg_file.replace('_segmentation.json', '_report.html')

    visualizer.generate_html(csv_file, seg_file, output_file)

    print(f"\nOpen {output_file} in your browser to view the report!")


if __name__ == "__main__":
    main()
