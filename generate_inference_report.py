"""
Generate Comprehensive Inference Report for VitalDB Model Testing

This script analyzes all inference results from case directories and generates
a detailed HTML report with visualizations and metrics.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')


def load_all_results(base_dir):
    """Load all prediction results from case directories"""

    pattern = os.path.join(base_dir, 'case_*', 'test_results_normalized', 'predictions.csv')
    result_files = glob.glob(pattern)

    print(f"Found {len(result_files)} result files")

    results = []
    for file_path in sorted(result_files):
        # Extract case number from path
        case_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        case_num = case_name.split('_')[1]

        # Load data
        df = pd.read_csv(file_path)
        df['case'] = case_name
        df['case_num'] = int(case_num)

        # Determine if in training set (cases 1-5)
        df['in_training_set'] = df['case_num'] <= 5

        results.append(df)

    if not results:
        raise ValueError("No result files found!")

    combined_df = pd.concat(results, ignore_index=True)
    print(f"Loaded {len(combined_df)} total predictions from {len(results)} cases")

    return combined_df, results


def compute_metrics(actual, predicted):
    """Compute comprehensive error metrics"""

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    # MARD (Mean Absolute Relative Difference)
    mard = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Median Absolute Error
    median_ae = np.median(np.abs(actual - predicted))

    # Percentage within thresholds
    errors = np.abs(actual - predicted)
    within_10 = np.sum(errors <= 10) / len(errors) * 100
    within_15 = np.sum(errors <= 15) / len(errors) * 100
    within_20 = np.sum(errors <= 20) / len(errors) * 100

    # Clarke Error Grid zones (simplified)
    zone_a, zone_b = compute_clarke_zones(actual, predicted)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R²': r2,
        'MARD (%)': mard,
        'Median AE': median_ae,
        'Within ±10 mg/dL (%)': within_10,
        'Within ±15 mg/dL (%)': within_15,
        'Within ±20 mg/dL (%)': within_20,
        'Clarke Zone A (%)': zone_a,
        'Clarke Zone B (%)': zone_b
    }


def compute_clarke_zones(actual, predicted):
    """Compute Clarke Error Grid zones A and B"""

    zone_a_count = 0
    zone_b_count = 0

    for ref, pred in zip(actual, predicted):
        # Zone A: Within 20% or both < 70
        if (ref < 70 and pred < 70) or (abs(pred - ref) <= 0.2 * ref):
            zone_a_count += 1
        # Zone B: Not in dangerous zones
        elif not (ref > 180 and pred < 70) and not (ref < 70 and pred > 180):
            zone_b_count += 1

    total = len(actual)
    zone_a_pct = (zone_a_count / total) * 100
    zone_b_pct = (zone_b_count / total) * 100

    return zone_a_pct, zone_b_pct


def create_plots(df, output_dir):
    """Generate all visualization plots"""

    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}

    # 1. Bland-Altman Plot
    plt.figure(figsize=(10, 6))
    mean_values = (df['actual_glucose_mg_dl'] + df['predicted_glucose_mg_dl']) / 2
    differences = df['predicted_glucose_mg_dl'] - df['actual_glucose_mg_dl']

    plt.scatter(mean_values, differences, alpha=0.3, s=10)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Agreement')
    plt.axhline(y=differences.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean Bias: {differences.mean():.2f} mg/dL')
    plt.axhline(y=differences.mean() + 1.96*differences.std(), color='gray', linestyle='--', label='±1.96 SD')
    plt.axhline(y=differences.mean() - 1.96*differences.std(), color='gray', linestyle='--')

    plt.xlabel('Mean Glucose (mg/dL)', fontsize=12, fontweight='bold')
    plt.ylabel('Difference (Predicted - Actual) mg/dL', fontsize=12, fontweight='bold')
    plt.title('Bland-Altman Plot: Agreement Between Predicted and Actual Glucose', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_files['bland_altman'] = os.path.join(output_dir, 'bland_altman.png')
    plt.savefig(plot_files['bland_altman'], dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Scatter Plot with Identity Line
    plt.figure(figsize=(10, 10))

    # Separate training and test data
    train_df = df[df['in_training_set']]
    test_df = df[~df['in_training_set']]

    plt.scatter(train_df['actual_glucose_mg_dl'], train_df['predicted_glucose_mg_dl'],
                alpha=0.4, s=20, label='Training Cases (1-5)', color='blue')
    plt.scatter(test_df['actual_glucose_mg_dl'], test_df['predicted_glucose_mg_dl'],
                alpha=0.4, s=20, label='Unseen Cases (6-10)', color='red')

    # Identity line
    min_val = min(df['actual_glucose_mg_dl'].min(), df['predicted_glucose_mg_dl'].min())
    max_val = max(df['actual_glucose_mg_dl'].max(), df['predicted_glucose_mg_dl'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Glucose (mg/dL)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Glucose (mg/dL)', fontsize=12, fontweight='bold')
    plt.title('Predicted vs Actual Glucose Levels', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plot_files['scatter'] = os.path.join(output_dir, 'scatter_plot.png')
    plt.savefig(plot_files['scatter'], dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Error Distribution by Case
    plt.figure(figsize=(14, 6))

    case_order = sorted(df['case'].unique(), key=lambda x: int(x.split('_')[1]))
    case_colors = ['blue' if int(c.split('_')[1]) <= 5 else 'red' for c in case_order]

    box_data = [df[df['case'] == case]['absolute_error_mg_dl'].values for case in case_order]
    bp = plt.boxplot(box_data, labels=[f"Case {c.split('_')[1]}" for c in case_order],
                     patch_artist=True, showmeans=True)

    for patch, color in zip(bp['boxes'], case_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='10 mg/dL (Excellent)')
    plt.axhline(y=15, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='15 mg/dL (Good)')
    plt.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='20 mg/dL (Fair)')

    plt.ylabel('Absolute Error (mg/dL)', fontsize=12, fontweight='bold')
    plt.xlabel('Case', fontsize=12, fontweight='bold')
    plt.title('Error Distribution by Case (Blue=Training Set, Red=Test Set)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_files['error_by_case'] = os.path.join(output_dir, 'error_by_case.png')
    plt.savefig(plot_files['error_by_case'], dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Cumulative Error Distribution
    plt.figure(figsize=(10, 6))

    train_errors = df[df['in_training_set']]['absolute_error_mg_dl'].values
    test_errors = df[~df['in_training_set']]['absolute_error_mg_dl'].values

    train_sorted = np.sort(train_errors)
    test_sorted = np.sort(test_errors)

    train_cumulative = np.arange(1, len(train_sorted) + 1) / len(train_sorted) * 100
    test_cumulative = np.arange(1, len(test_sorted) + 1) / len(test_sorted) * 100

    plt.plot(train_sorted, train_cumulative, linewidth=2, label='Training Cases (1-5)', color='blue')
    plt.plot(test_sorted, test_cumulative, linewidth=2, label='Unseen Cases (6-10)', color='red')

    plt.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='10 mg/dL')
    plt.axvline(x=15, color='orange', linestyle='--', alpha=0.7, label='15 mg/dL')
    plt.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='20 mg/dL')

    plt.xlabel('Absolute Error (mg/dL)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_files['cumulative_error'] = os.path.join(output_dir, 'cumulative_error.png')
    plt.savefig(plot_files['cumulative_error'], dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Per-Case Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    case_metrics = []
    for case in case_order:
        case_df = df[df['case'] == case]
        metrics = compute_metrics(case_df['actual_glucose_mg_dl'], case_df['predicted_glucose_mg_dl'])
        metrics['Case'] = f"Case {case.split('_')[1]}"
        metrics['Type'] = 'Training' if int(case.split('_')[1]) <= 5 else 'Test'
        case_metrics.append(metrics)

    metrics_df = pd.DataFrame(case_metrics)

    # MAE
    axes[0, 0].bar(range(len(metrics_df)), metrics_df['MAE'],
                   color=['blue' if t == 'Training' else 'red' for t in metrics_df['Type']])
    axes[0, 0].set_xticks(range(len(metrics_df)))
    axes[0, 0].set_xticklabels(metrics_df['Case'], rotation=45)
    axes[0, 0].set_ylabel('MAE (mg/dL)', fontweight='bold')
    axes[0, 0].set_title('Mean Absolute Error by Case', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # RMSE
    axes[0, 1].bar(range(len(metrics_df)), metrics_df['RMSE'],
                   color=['blue' if t == 'Training' else 'red' for t in metrics_df['Type']])
    axes[0, 1].set_xticks(range(len(metrics_df)))
    axes[0, 1].set_xticklabels(metrics_df['Case'], rotation=45)
    axes[0, 1].set_ylabel('RMSE (mg/dL)', fontweight='bold')
    axes[0, 1].set_title('Root Mean Squared Error by Case', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # R²
    axes[1, 0].bar(range(len(metrics_df)), metrics_df['R²'],
                   color=['blue' if t == 'Training' else 'red' for t in metrics_df['Type']])
    axes[1, 0].set_xticks(range(len(metrics_df)))
    axes[1, 0].set_xticklabels(metrics_df['Case'], rotation=45)
    axes[1, 0].set_ylabel('R² Score', fontweight='bold')
    axes[1, 0].set_title('R² Score by Case', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Clarke Zone A
    axes[1, 1].bar(range(len(metrics_df)), metrics_df['Clarke Zone A (%)'],
                   color=['blue' if t == 'Training' else 'red' for t in metrics_df['Type']])
    axes[1, 1].set_xticks(range(len(metrics_df)))
    axes[1, 1].set_xticklabels(metrics_df['Case'], rotation=45)
    axes[1, 1].set_ylabel('Clarke Zone A (%)', fontweight='bold')
    axes[1, 1].set_title('Clinical Accuracy (Clarke Zone A) by Case', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=75, color='orange', linestyle='--', label='75% threshold')
    axes[1, 1].legend()

    plt.tight_layout()
    plot_files['metrics_comparison'] = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(plot_files['metrics_comparison'], dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated {len(plot_files)} plots in {output_dir}")
    return plot_files


def generate_html_report(df, output_file):
    """Generate comprehensive HTML report"""

    # Compute overall metrics
    all_metrics = compute_metrics(df['actual_glucose_mg_dl'], df['predicted_glucose_mg_dl'])

    # Compute metrics for training vs test sets
    train_df = df[df['in_training_set']]
    test_df = df[~df['in_training_set']]

    train_metrics = compute_metrics(train_df['actual_glucose_mg_dl'], train_df['predicted_glucose_mg_dl'])
    test_metrics = compute_metrics(test_df['actual_glucose_mg_dl'], test_df['predicted_glucose_mg_dl'])

    # Per-case metrics
    case_metrics_list = []
    for case in sorted(df['case'].unique(), key=lambda x: int(x.split('_')[1])):
        case_df = df[df['case'] == case]
        metrics = compute_metrics(case_df['actual_glucose_mg_dl'], case_df['predicted_glucose_mg_dl'])
        metrics['Case'] = case
        metrics['Case Number'] = int(case.split('_')[1])
        metrics['In Training'] = 'Yes' if case_df['in_training_set'].iloc[0] else 'No'
        metrics['Samples'] = len(case_df)
        case_metrics_list.append(metrics)

    # Generate plots
    plot_dir = os.path.join(os.path.dirname(output_file), 'plots')
    plot_files = create_plots(df, plot_dir)

    # Create HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VitalDB Model Inference Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
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
            padding: 40px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .report-date {{
            font-size: 0.9em;
            margin-top: 10px;
            opacity: 0.8;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section-title {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .subsection-title {{
            font-size: 1.5em;
            color: #764ba2;
            margin: 30px 0 15px 0;
        }}

        .highlight-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .warning-box {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .success-box {{
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}

        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}

        tr:hover {{
            background-color: #e9ecef;
        }}

        .metric-value {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }}

        .training-case {{
            background-color: #cfe2ff !important;
        }}

        .test-case {{
            background-color: #f8d7da !important;
        }}

        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}

        .plot-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .metric-card {{
            background: white;
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}

        .metric-card-title {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}

        .metric-card-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}

        .metric-card-unit {{
            font-size: 0.8em;
            color: #999;
        }}

        .comparison-table {{
            margin: 30px 0;
        }}

        .excellent {{ color: #28a745; font-weight: bold; }}
        .good {{ color: #17a2b8; font-weight: bold; }}
        .fair {{ color: #ffc107; font-weight: bold; }}
        .poor {{ color: #dc3545; font-weight: bold; }}

        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}

        .legend {{
            display: inline-block;
            margin: 10px 20px;
        }}

        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}

        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border-radius: 3px;
        }}

        .training-color {{ background-color: #0d6efd; }}
        .test-color {{ background-color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 VitalDB Blood Glucose Prediction Model</h1>
            <h2>Comprehensive Inference Report</h2>
            <p>ResNet34 Deep Learning Model - PPG Signal Analysis</p>
            <p class="report-date">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </header>

        <div class="content">
            <!-- Executive Summary -->
            <div class="section">
                <h2 class="section-title">📊 Executive Summary</h2>

                <div class="highlight-box">
                    <h3>Dataset Overview</h3>
                    <p><strong>Total Predictions:</strong> {len(df):,} glucose estimations</p>
                    <p><strong>Cases Analyzed:</strong> {len(df['case'].unique())} patient cases</p>
                    <p><strong>Training Set Cases:</strong> Cases 1-5 ({len(train_df):,} predictions)</p>
                    <p><strong>Test Set Cases (Unseen):</strong> Cases 6-10 ({len(test_df):,} predictions)</p>
                </div>

                <div class="legend">
                    <div class="legend-item">
                        <span class="legend-color training-color"></span>
                        Training Cases (1-5)
                    </div>
                    <div class="legend-item">
                        <span class="legend-color test-color"></span>
                        Test Cases (6-10) - Not in Training
                    </div>
                </div>

                <h3 class="subsection-title">Overall Model Performance</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-card-title">Mean Absolute Error</div>
                        <div class="metric-card-value">{all_metrics['MAE']:.2f}</div>
                        <div class="metric-card-unit">mg/dL</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-title">Root Mean Squared Error</div>
                        <div class="metric-card-value">{all_metrics['RMSE']:.2f}</div>
                        <div class="metric-card-unit">mg/dL</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-title">R² Score</div>
                        <div class="metric-card-value">{all_metrics['R²']:.4f}</div>
                        <div class="metric-card-unit"></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-title">MARD</div>
                        <div class="metric-card-value">{all_metrics['MARD (%)']:.2f}</div>
                        <div class="metric-card-unit">%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-title">Clarke Zone A</div>
                        <div class="metric-card-value">{all_metrics['Clarke Zone A (%)']:.1f}</div>
                        <div class="metric-card-unit">%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-title">Within ±20 mg/dL</div>
                        <div class="metric-card-value">{all_metrics['Within ±20 mg/dL (%)']:.1f}</div>
                        <div class="metric-card-unit">%</div>
                    </div>
                </div>

                <div class="{'success-box' if all_metrics['MAE'] < 15 else 'warning-box' if all_metrics['MAE'] < 20 else 'highlight-box'}">
                    <strong>Clinical Assessment:</strong>
                    {
                        'EXCELLENT - MAE < 15 mg/dL indicates high clinical accuracy' if all_metrics['MAE'] < 15
                        else 'GOOD - MAE < 20 mg/dL indicates acceptable clinical accuracy' if all_metrics['MAE'] < 20
                        else 'FAIR - MAE > 20 mg/dL suggests room for improvement'
                    }
                </div>
            </div>

            <!-- Training vs Test Comparison -->
            <div class="section">
                <h2 class="section-title">🔬 Training vs Test Set Comparison</h2>

                <div class="warning-box">
                    <strong>⚠️ Important Note:</strong> Cases 1-5 were part of the model training dataset,
                    while Cases 6-10 are completely unseen data not used during training. Comparing these
                    metrics helps assess model generalization capability.
                </div>

                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Training Cases (1-5)</th>
                            <th>Test Cases (6-10)</th>
                            <th>Difference</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Number of Samples</strong></td>
                            <td>{len(train_df):,}</td>
                            <td>{len(test_df):,}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td><strong>MAE (mg/dL)</strong></td>
                            <td class="metric-value">{train_metrics['MAE']:.2f}</td>
                            <td class="metric-value">{test_metrics['MAE']:.2f}</td>
                            <td>{test_metrics['MAE'] - train_metrics['MAE']:+.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>RMSE (mg/dL)</strong></td>
                            <td class="metric-value">{train_metrics['RMSE']:.2f}</td>
                            <td class="metric-value">{test_metrics['RMSE']:.2f}</td>
                            <td>{test_metrics['RMSE'] - train_metrics['RMSE']:+.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>R² Score</strong></td>
                            <td class="metric-value">{train_metrics['R²']:.4f}</td>
                            <td class="metric-value">{test_metrics['R²']:.4f}</td>
                            <td>{test_metrics['R²'] - train_metrics['R²']:+.4f}</td>
                        </tr>
                        <tr>
                            <td><strong>MARD (%)</strong></td>
                            <td class="metric-value">{train_metrics['MARD (%)']:.2f}%</td>
                            <td class="metric-value">{test_metrics['MARD (%)']:.2f}%</td>
                            <td>{test_metrics['MARD (%)'] - train_metrics['MARD (%)']:+.2f}%</td>
                        </tr>
                        <tr>
                            <td><strong>Clarke Zone A (%)</strong></td>
                            <td class="metric-value">{train_metrics['Clarke Zone A (%)']:.1f}%</td>
                            <td class="metric-value">{test_metrics['Clarke Zone A (%)']:.1f}%</td>
                            <td>{test_metrics['Clarke Zone A (%)'] - train_metrics['Clarke Zone A (%)']:+.1f}%</td>
                        </tr>
                        <tr>
                            <td><strong>Within ±10 mg/dL</strong></td>
                            <td class="metric-value">{train_metrics['Within ±10 mg/dL (%)']:.1f}%</td>
                            <td class="metric-value">{test_metrics['Within ±10 mg/dL (%)']:.1f}%</td>
                            <td>{test_metrics['Within ±10 mg/dL (%)'] - train_metrics['Within ±10 mg/dL (%)']:+.1f}%</td>
                        </tr>
                        <tr>
                            <td><strong>Within ±15 mg/dL</strong></td>
                            <td class="metric-value">{train_metrics['Within ±15 mg/dL (%)']:.1f}%</td>
                            <td class="metric-value">{test_metrics['Within ±15 mg/dL (%)']:.1f}%</td>
                            <td>{test_metrics['Within ±15 mg/dL (%)'] - train_metrics['Within ±15 mg/dL (%)']:+.1f}%</td>
                        </tr>
                        <tr>
                            <td><strong>Within ±20 mg/dL</strong></td>
                            <td class="metric-value">{train_metrics['Within ±20 mg/dL (%)']:.1f}%</td>
                            <td class="metric-value">{test_metrics['Within ±20 mg/dL (%)']:.1f}%</td>
                            <td>{test_metrics['Within ±20 mg/dL (%)'] - train_metrics['Within ±20 mg/dL (%)']:+.1f}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Per-Case Analysis -->
            <div class="section">
                <h2 class="section-title">📋 Individual Case Performance</h2>

                <table>
                    <thead>
                        <tr>
                            <th>Case</th>
                            <th>In Training</th>
                            <th>Samples</th>
                            <th>MAE (mg/dL)</th>
                            <th>RMSE (mg/dL)</th>
                            <th>R²</th>
                            <th>MARD (%)</th>
                            <th>Zone A (%)</th>
                            <th>±20 mg/dL (%)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Add per-case rows
    for metric in case_metrics_list:
        row_class = 'training-case' if metric['In Training'] == 'Yes' else 'test-case'
        mae_class = 'excellent' if metric['MAE'] < 10 else 'good' if metric['MAE'] < 15 else 'fair' if metric['MAE'] < 20 else 'poor'

        html_content += f"""
                        <tr class="{row_class}">
                            <td><strong>Case {metric['Case Number']}</strong></td>
                            <td>{metric['In Training']}</td>
                            <td>{metric['Samples']:,}</td>
                            <td class="{mae_class}">{metric['MAE']:.2f}</td>
                            <td>{metric['RMSE']:.2f}</td>
                            <td>{metric['R²']:.4f}</td>
                            <td>{metric['MARD (%)']:.2f}</td>
                            <td>{metric['Clarke Zone A (%)']:.1f}</td>
                            <td>{metric['Within ±20 mg/dL (%)']:.1f}</td>
                        </tr>
"""

    html_content += """
                    </tbody>
                </table>
            </div>

            <!-- Visualizations -->
            <div class="section">
                <h2 class="section-title">📈 Visualizations</h2>
"""

    # Add scatter plot
    if 'scatter' in plot_files:
        html_content += f"""
                <h3 class="subsection-title">Predicted vs Actual Glucose Levels</h3>
                <div class="plot-container">
                    <img src="plots/{os.path.basename(plot_files['scatter'])}" alt="Scatter Plot">
                    <p class="plot-caption">
                        Blue points represent training cases (1-5), red points represent unseen test cases (6-10).
                        Points closer to the diagonal line indicate better predictions.
                    </p>
                </div>
"""

    # Add Bland-Altman plot
    if 'bland_altman' in plot_files:
        html_content += f"""
                <h3 class="subsection-title">Bland-Altman Agreement Plot</h3>
                <div class="plot-container">
                    <img src="plots/{os.path.basename(plot_files['bland_altman'])}" alt="Bland-Altman Plot">
                    <p class="plot-caption">
                        Shows the agreement between predicted and actual values. Points within ±1.96 SD
                        indicate acceptable agreement. Mean bias shows systematic over/under-estimation.
                    </p>
                </div>
"""

    # Add error by case plot
    if 'error_by_case' in plot_files:
        html_content += f"""
                <h3 class="subsection-title">Error Distribution by Case</h3>
                <div class="plot-container">
                    <img src="plots/{os.path.basename(plot_files['error_by_case'])}" alt="Error by Case">
                    <p class="plot-caption">
                        Box plots show the distribution of absolute errors for each case. Blue boxes = training cases,
                        red boxes = test cases. Lower boxes indicate better performance.
                    </p>
                </div>
"""

    # Add cumulative error plot
    if 'cumulative_error' in plot_files:
        html_content += f"""
                <h3 class="subsection-title">Cumulative Error Distribution</h3>
                <div class="plot-container">
                    <img src="plots/{os.path.basename(plot_files['cumulative_error'])}" alt="Cumulative Error">
                    <p class="plot-caption">
                        Shows what percentage of predictions fall within different error thresholds.
                        Steeper curves indicate more predictions with lower errors.
                    </p>
                </div>
"""

    # Add metrics comparison plot
    if 'metrics_comparison' in plot_files:
        html_content += f"""
                <h3 class="subsection-title">Comprehensive Metrics Comparison</h3>
                <div class="plot-container">
                    <img src="plots/{os.path.basename(plot_files['metrics_comparison'])}" alt="Metrics Comparison">
                    <p class="plot-caption">
                        Comparison of key performance metrics across all cases. Blue bars = training cases,
                        red bars = test cases.
                    </p>
                </div>
"""

    html_content += """
            </div>

            <!-- Key Findings -->
            <div class="section">
                <h2 class="section-title">🔍 Key Findings</h2>
"""

    # Generate key findings
    findings = []

    # Overall performance
    if all_metrics['MAE'] < 15:
        findings.append(f"✅ <strong>Excellent Overall Performance:</strong> MAE of {all_metrics['MAE']:.2f} mg/dL indicates clinically acceptable accuracy.")
    elif all_metrics['MAE'] < 20:
        findings.append(f"✓ <strong>Good Overall Performance:</strong> MAE of {all_metrics['MAE']:.2f} mg/dL is within acceptable range.")
    else:
        findings.append(f"⚠️ <strong>Fair Overall Performance:</strong> MAE of {all_metrics['MAE']:.2f} mg/dL suggests room for improvement.")

    # Clarke zone
    if all_metrics['Clarke Zone A (%)'] > 75:
        findings.append(f"✅ <strong>High Clinical Accuracy:</strong> {all_metrics['Clarke Zone A (%)']:.1f}% of predictions in Clarke Error Grid Zone A.")

    # Generalization
    mae_diff = test_metrics['MAE'] - train_metrics['MAE']
    if abs(mae_diff) < 5:
        findings.append(f"✅ <strong>Excellent Generalization:</strong> Test set MAE ({test_metrics['MAE']:.2f} mg/dL) is very close to training set MAE ({train_metrics['MAE']:.2f} mg/dL), indicating good model generalization to unseen data.")
    elif abs(mae_diff) < 10:
        findings.append(f"✓ <strong>Good Generalization:</strong> Test set MAE ({test_metrics['MAE']:.2f} mg/dL) vs training set MAE ({train_metrics['MAE']:.2f} mg/dL) shows reasonable generalization.")
    else:
        findings.append(f"⚠️ <strong>Generalization Gap:</strong> Test set MAE ({test_metrics['MAE']:.2f} mg/dL) differs significantly from training set MAE ({train_metrics['MAE']:.2f} mg/dL), suggesting potential overfitting.")

    # Error thresholds
    findings.append(f"📊 <strong>Error Distribution:</strong> {all_metrics['Within ±10 mg/dL (%)']:.1f}% within ±10 mg/dL, {all_metrics['Within ±15 mg/dL (%)']:.1f}% within ±15 mg/dL, {all_metrics['Within ±20 mg/dL (%)']:.1f}% within ±20 mg/dL.")

    # R² score
    if all_metrics['R²'] > 0.7:
        findings.append(f"✅ <strong>Strong Correlation:</strong> R² score of {all_metrics['R²']:.4f} indicates strong predictive power.")
    elif all_metrics['R²'] > 0.5:
        findings.append(f"✓ <strong>Moderate Correlation:</strong> R² score of {all_metrics['R²']:.4f} indicates moderate predictive power.")
    else:
        findings.append(f"⚠️ <strong>Weak Correlation:</strong> R² score of {all_metrics['R²']:.4f} suggests limited predictive power.")

    for finding in findings:
        html_content += f"""
                <div class="highlight-box">
                    <p>{finding}</p>
                </div>
"""

    html_content += """
            </div>

            <!-- Methodology -->
            <div class="section">
                <h2 class="section-title">🔬 Methodology</h2>

                <h3 class="subsection-title">Model Architecture</h3>
                <div class="highlight-box">
                    <p><strong>Model:</strong> ResNet34 adapted for 1D PPG signal analysis</p>
                    <p><strong>Input:</strong> 1-second PPG signal segments (100 Hz sampling rate)</p>
                    <p><strong>Output:</strong> Blood glucose level prediction (mg/dL)</p>
                    <p><strong>Training Data:</strong> VitalDB dataset with perioperative biosignal data</p>
                </div>

                <h3 class="subsection-title">Evaluation Metrics</h3>
                <ul style="line-height: 2;">
                    <li><strong>MAE (Mean Absolute Error):</strong> Average absolute difference between predicted and actual values</li>
                    <li><strong>RMSE (Root Mean Squared Error):</strong> Square root of average squared differences (penalizes larger errors more)</li>
                    <li><strong>R² Score:</strong> Proportion of variance explained by the model (closer to 1 is better)</li>
                    <li><strong>MARD (Mean Absolute Relative Difference):</strong> Average relative error as percentage</li>
                    <li><strong>Clarke Error Grid Analysis:</strong> Clinical accuracy assessment (Zone A = clinically accurate)</li>
                    <li><strong>Error Thresholds:</strong> Percentage of predictions within ±10, ±15, and ±20 mg/dL</li>
                </ul>

                <h3 class="subsection-title">Dataset Split</h3>
                <div class="highlight-box">
                    <p><strong>Training Cases (1-5):</strong> Cases used during model training - expected to have better performance</p>
                    <p><strong>Test Cases (6-10):</strong> Completely unseen cases not used in training - tests generalization ability</p>
                    <p><strong>Importance:</strong> Similar performance between training and test sets indicates good generalization and reduced overfitting</p>
                </div>
            </div>

            <!-- Recommendations -->
            <div class="section">
                <h2 class="section-title">💡 Recommendations</h2>
"""

    recommendations = []

    if test_metrics['MAE'] > train_metrics['MAE'] + 5:
        recommendations.append("Consider collecting more diverse training data to improve generalization to unseen cases.")

    if all_metrics['MAE'] > 20:
        recommendations.append("Focus on improving model architecture or feature engineering to reduce MAE below 20 mg/dL.")

    if all_metrics['Clarke Zone A (%)'] < 75:
        recommendations.append("Clinical accuracy could be improved - target >75% in Clarke Zone A for clinical deployment.")

    if all_metrics['R²'] < 0.5:
        recommendations.append("Low R² score suggests the model may benefit from additional input features or different architecture.")

    recommendations.append("Continue validating the model on additional independent datasets before clinical deployment.")
    recommendations.append("Consider implementing ensemble methods or model averaging to improve robustness.")
    recommendations.append("Investigate cases with highest errors to identify patterns and potential data quality issues.")

    for i, rec in enumerate(recommendations, 1):
        html_content += f"""
                <div class="highlight-box">
                    <p><strong>{i}.</strong> {rec}</p>
                </div>
"""

    html_content += f"""
            </div>
        </div>

        <footer>
            <p><strong>VitalDB Blood Glucose Prediction Model - Inference Report</strong></p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>Model: ResNet34 1D CNN | Framework: PyTorch | Dataset: VitalDB</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                For questions or technical details, please refer to the project documentation.
            </p>
        </footer>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n[OK] HTML report generated: {output_file}")
    print(f"[OK] Plots saved in: {plot_dir}")


def main():
    """Main execution function"""

    print("=" * 80)
    print("VitalDB Model Inference Report Generator")
    print("=" * 80)

    # Configuration
    base_dir = r'C:\IITM\vitalDB\data\web_app_data'
    output_file = r'C:\IITM\vitalDB\data\inference_report.html'

    try:
        # Load all results
        print("\n[*] Loading inference results...")
        combined_df, individual_dfs = load_all_results(base_dir)

        # Generate report
        print("\n[*] Generating HTML report...")
        generate_html_report(combined_df, output_file)

        print("\n" + "=" * 80)
        print("[OK] REPORT GENERATION COMPLETED")
        print("=" * 80)
        print(f"\n[*] Open the report in your browser:")
        print(f"   {output_file}")

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
