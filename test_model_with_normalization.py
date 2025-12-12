"""
Test Trained Model With Proper Normalization Handling

This script correctly handles the glucose normalization/denormalization
that was used during training.

Usage:
    python test_model_with_normalization.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.resnet34_glucose_predictor import ResNet34_1D


def load_test_data(data_dir):
    """Load PPG windows and glucose labels from CSV files"""

    print("=" * 80)
    print("Loading Test Data")
    print("=" * 80)

    # Load PPG windows
    ppg_file = os.path.join(data_dir, 'ppg_windows.csv')
    print(f"Loading PPG windows from: {ppg_file}")
    ppg_df = pd.read_csv(ppg_file)

    # Group by window_index and check lengths
    windows = []
    window_lengths = []
    for window_idx in sorted(ppg_df['window_index'].unique()):
        window_df = ppg_df[ppg_df['window_index'] == window_idx].sort_values('sample_index')
        window = window_df['amplitude'].values
        windows.append(window)
        window_lengths.append(len(window))

    # Find the most common window length
    from collections import Counter
    length_counts = Counter(window_lengths)
    target_length = length_counts.most_common(1)[0][0]

    print(f"Window length statistics:")
    print(f"  Min length: {min(window_lengths)}")
    print(f"  Max length: {max(window_lengths)}")
    print(f"  Target length: {target_length}")
    print(f"  Total windows: {len(windows)}")

    # Pad or truncate windows to target length
    normalized_windows = []
    for window in windows:
        if len(window) == target_length:
            normalized_windows.append(window)
        elif len(window) > target_length:
            # Truncate
            normalized_windows.append(window[:target_length])
        else:
            # Pad with zeros
            padded = np.zeros(target_length)
            padded[:len(window)] = window
            normalized_windows.append(padded)

    ppg_data = np.array(normalized_windows)
    print(f"[OK] Loaded {len(ppg_data)} PPG windows")
    print(f"  Shape: {ppg_data.shape}")

    # Load glucose labels
    glucose_file = os.path.join(data_dir, 'glucose_labels.csv')
    print(f"\nLoading glucose labels from: {glucose_file}")
    glucose_df = pd.read_csv(glucose_file)
    glucose_data = glucose_df['glucose_mg_dl'].values

    print(f"[OK] Loaded {len(glucose_data)} glucose labels")
    print(f"  Range: {glucose_data.min():.1f} - {glucose_data.max():.1f} mg/dL")
    print(f"  Mean: {glucose_data.mean():.1f} mg/dL")

    return ppg_data, glucose_data


def normalize_ppg(ppg_data):
    """Normalize PPG data (per-window normalization)"""
    ppg_mean = np.mean(ppg_data, axis=1, keepdims=True)
    ppg_std = np.std(ppg_data, axis=1, keepdims=True)
    ppg_std[ppg_std == 0] = 1.0  # Avoid division by zero
    normalized_ppg = (ppg_data - ppg_mean) / ppg_std
    return normalized_ppg


def normalize_glucose(glucose_data):
    """
    Normalize glucose data using the same method as training.
    Returns normalized glucose, mean, and std for denormalization.
    """
    glucose_mean = np.mean(glucose_data)
    glucose_std = np.std(glucose_data)

    # Handle constant glucose values (std = 0)
    if glucose_std == 0:
        glucose_std = 1.0
        normalized_glucose = glucose_data - glucose_mean
    else:
        normalized_glucose = (glucose_data - glucose_mean) / glucose_std

    return normalized_glucose, glucose_mean, glucose_std


def denormalize_glucose(normalized_glucose, glucose_mean, glucose_std):
    """Convert normalized glucose back to mg/dL"""
    return normalized_glucose * glucose_std + glucose_mean


def test_model_with_normalization(model_path, test_data_dir):
    """
    Test model with proper normalization handling
    """
    print("\n" + "=" * 80)
    print("TESTING MODEL WITH PROPER NORMALIZATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test Data: {test_data_dir}")

    # Load data
    ppg_data, glucose_data = load_test_data(test_data_dir)

    # Normalize data (same as training)
    print("\n" + "=" * 80)
    print("Normalizing Data")
    print("=" * 80)

    ppg_normalized = normalize_ppg(ppg_data)
    glucose_normalized, glucose_mean, glucose_std = normalize_glucose(glucose_data)

    print(f"Glucose normalization:")
    print(f"  Mean: {glucose_mean:.2f} mg/dL")
    print(f"  Std: {glucose_std:.2f} mg/dL")
    print(f"  Normalized range: {glucose_normalized.min():.4f} - {glucose_normalized.max():.4f}")

    # Load model
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    input_length = ppg_data.shape[1]
    model = ResNet34_1D(input_length=input_length, num_classes=1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded successfully")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Training metrics: {checkpoint.get('metrics', 'N/A')}")

    # Run predictions
    print("\n" + "=" * 80)
    print("Running Predictions")
    print("=" * 80)

    all_predictions_normalized = []

    batch_size = 32
    num_samples = len(ppg_normalized)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Get batch (handles last batch correctly)
            end_idx = min(i + batch_size, num_samples)
            batch_ppg = ppg_normalized[i:end_idx]

            # Convert to tensor and add channel dimension
            batch_tensor = torch.tensor(batch_ppg, dtype=torch.float32).unsqueeze(1).to(device)

            # Predict (outputs normalized values)
            predictions = model(batch_tensor)

            # Append predictions (ensure we get the correct number)
            batch_predictions = predictions.cpu().numpy().flatten()
            all_predictions_normalized.extend(batch_predictions)

    predictions_normalized = np.array(all_predictions_normalized)

    print(f"[OK] Generated {len(predictions_normalized)} predictions")
    print(f"  Expected: {num_samples} samples")
    print(f"  Normalized predictions range: {predictions_normalized.min():.4f} - {predictions_normalized.max():.4f}")

    # Denormalize predictions
    print("\n" + "=" * 80)
    print("Denormalizing Predictions")
    print("=" * 80)

    predictions_mgdl = denormalize_glucose(predictions_normalized, glucose_mean, glucose_std)
    actuals_mgdl = glucose_data

    print(f"Denormalized predictions:")
    print(f"  Range: {predictions_mgdl.min():.2f} - {predictions_mgdl.max():.2f} mg/dL")
    print(f"  Mean: {predictions_mgdl.mean():.2f} mg/dL")
    print(f"  Std: {predictions_mgdl.std():.2f} mg/dL")

    # Ensure arrays have matching lengths
    min_length = min(len(predictions_mgdl), len(actuals_mgdl))
    if len(predictions_mgdl) != len(actuals_mgdl):
        print(f"\n[WARNING] Length mismatch detected:")
        print(f"  Predictions: {len(predictions_mgdl)}")
        print(f"  Actuals: {len(actuals_mgdl)}")
        print(f"  Truncating to: {min_length}")
        predictions_mgdl = predictions_mgdl[:min_length]
        actuals_mgdl = actuals_mgdl[:min_length]

    # Compute metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    mae = mean_absolute_error(actuals_mgdl, predictions_mgdl)
    mse = mean_squared_error(actuals_mgdl, predictions_mgdl)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals_mgdl, predictions_mgdl)

    print(f"\nRegression Metrics:")
    print(f"  MAE:  {mae:.2f} mg/dL")
    print(f"  RMSE: {rmse:.2f} mg/dL")
    print(f"  MSE:  {mse:.2f}")
    print(f"  R²:   {r2:.4f}")

    print(f"\nPrediction vs Actual:")
    print(f"  Predicted: {predictions_mgdl.mean():.2f} ± {predictions_mgdl.std():.2f} mg/dL")
    print(f"  Actual:    {actuals_mgdl.mean():.2f} ± {actuals_mgdl.std():.2f} mg/dL")

    # Clinical interpretation
    print("\nClinical Interpretation:")
    if mae < 10:
        print("  [OK] EXCELLENT: MAE < 10 mg/dL")
    elif mae < 15:
        print("  [OK] GOOD: MAE < 15 mg/dL")
    elif mae < 20:
        print("  [WARN] FAIR: MAE < 20 mg/dL")
    else:
        print("  [ERROR] POOR: MAE > 20 mg/dL")

    # Show sample predictions
    print("\nSample Predictions (first 20):")
    print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 48)
    for i in range(min(20, len(predictions_mgdl))):
        error = predictions_mgdl[i] - actuals_mgdl[i]
        print(f"{i:<8} {actuals_mgdl[i]:<12.2f} {predictions_mgdl[i]:<12.2f} {error:<12.2f}")

    # Save results
    output_dir = os.path.join(test_data_dir, 'test_results_normalized')
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame({
        'window_index': range(len(predictions_mgdl)),
        'actual_glucose_mg_dl': actuals_mgdl,
        'predicted_glucose_mg_dl': predictions_mgdl,
        'error_mg_dl': predictions_mgdl - actuals_mgdl,
        'absolute_error_mg_dl': np.abs(predictions_mgdl - actuals_mgdl)
    })

    output_file = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n[OK] Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained model with proper normalization')
    parser.add_argument('--model_path', type=str,
                       default=r'C:\IITM\vitalDB\model\best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default=r'C:\IITM\vitalDB\data\web_app_data\case_1_SNUADC_PLETH',
                       help='Path to test data directory')

    args = parser.parse_args()

    test_model_with_normalization(args.model_path, args.test_data)
