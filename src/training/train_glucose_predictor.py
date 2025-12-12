"""
Training script for ResNet34-1D Glucose Prediction Model

This script trains the ResNet34-1D model on paired PPG-glucose data.
It includes:
- Data loading and preprocessing
- Training loop with validation
- Model checkpointing
- Performance metrics tracking
- TensorBoard logging
- Early stopping

Usage:
    python train_glucose_predictor.py --data_dir <path_to_data> --epochs 100 --batch_size 32

Requirements:
    - Paired PPG-glucose data in CSV format
    - GPU with 8+ GB VRAM (recommended)
    - PyTorch with CUDA support
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from src.training.resnet34_glucose_predictor import ResNet34_1D


class PPGGlucoseDataset(Dataset):
    """
    PyTorch Dataset for paired PPG windows and glucose values

    Expected data format:
    - ppg_data: numpy array of shape (N, window_length) - N PPG windows
    - glucose_data: numpy array of shape (N,) - N glucose values in mg/dL
    """

    def __init__(self, ppg_data, glucose_data, normalize=True):
        """
        Args:
            ppg_data: numpy array (N, window_length)
            glucose_data: numpy array (N,)
            normalize: whether to normalize PPG signals
        """
        self.ppg_data = ppg_data
        self.glucose_data = glucose_data
        self.normalize = normalize

        # Normalize PPG signals if requested
        if self.normalize:
            self.ppg_mean = np.mean(ppg_data, axis=1, keepdims=True)
            self.ppg_std = np.std(ppg_data, axis=1, keepdims=True)
            self.ppg_std[self.ppg_std == 0] = 1.0  # Avoid division by zero
            self.ppg_data = (ppg_data - self.ppg_mean) / self.ppg_std

        # Normalize glucose values (optional - helps training stability)
        self.glucose_mean = np.mean(glucose_data)
        self.glucose_std = np.std(glucose_data)

        # Handle constant glucose values (std = 0)
        if self.glucose_std == 0:
            self.glucose_std = 1.0
            self.normalized_glucose = glucose_data - self.glucose_mean
        else:
            self.normalized_glucose = (glucose_data - self.glucose_mean) / self.glucose_std

        print(f"Dataset initialized:")
        print(f"  PPG windows: {len(ppg_data)}")
        print(f"  Window length: {ppg_data.shape[1]} samples")
        print(f"  Glucose range: {glucose_data.min():.1f} - {glucose_data.max():.1f} mg/dL")
        print(f"  Glucose mean: {self.glucose_mean:.1f} mg/dL")
        print(f"  Glucose std: {self.glucose_std:.1f} mg/dL")

    def __len__(self):
        return len(self.ppg_data)

    def __getitem__(self, idx):
        """
        Returns:
            ppg: tensor of shape (1, window_length)
            glucose: tensor of shape (1,)
        """
        ppg = torch.tensor(self.ppg_data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        glucose = torch.tensor(self.normalized_glucose[idx], dtype=torch.float32).unsqueeze(0)

        return ppg, glucose

    def denormalize_glucose(self, normalized_glucose):
        """Convert normalized glucose back to mg/dL"""
        return normalized_glucose * self.glucose_std + self.glucose_mean


def load_data_from_csv(data_dir):
    """
    Load paired PPG-glucose data from CSV files

    Expected files in data_dir:
    - ppg_windows.csv: columns = [window_index, sample_index, amplitude]
    - glucose_labels.csv: columns = [window_index, glucose_mg_dl]

    Returns:
        ppg_data: numpy array (N, window_length)
        glucose_data: numpy array (N,)
    """
    ppg_file = os.path.join(data_dir, 'ppg_windows.csv')
    glucose_file = os.path.join(data_dir, 'glucose_labels.csv')

    if not os.path.exists(ppg_file):
        raise FileNotFoundError(f"PPG data file not found: {ppg_file}")
    if not os.path.exists(glucose_file):
        raise FileNotFoundError(f"Glucose labels file not found: {glucose_file}")

    # Load PPG windows
    print(f"Loading PPG data from {ppg_file}...")
    ppg_df = pd.read_csv(ppg_file)

    # Group by window_index and reconstruct windows
    windows = []
    window_lengths = []
    for window_idx in sorted(ppg_df['window_index'].unique()):
        window_df = ppg_df[ppg_df['window_index'] == window_idx].sort_values('sample_index')
        window = window_df['amplitude'].values
        windows.append(window)
        window_lengths.append(len(window))

    # Check if all windows have the same length
    unique_lengths = set(window_lengths)
    if len(unique_lengths) > 1:
        print(f"WARNING: Windows have different lengths: {unique_lengths}")
        print(f"  Min length: {min(window_lengths)}")
        print(f"  Max length: {max(window_lengths)}")
        print(f"  Most common length: {max(set(window_lengths), key=window_lengths.count)}")

        # Use the most common length as target
        target_length = max(set(window_lengths), key=window_lengths.count)
        print(f"  Normalizing all windows to length: {target_length}")

        # Pad or truncate windows to target length
        normalized_windows = []
        for window in windows:
            if len(window) < target_length:
                # Pad with zeros
                padded = np.pad(window, (0, target_length - len(window)), mode='constant', constant_values=0)
                normalized_windows.append(padded)
            elif len(window) > target_length:
                # Truncate
                normalized_windows.append(window[:target_length])
            else:
                normalized_windows.append(window)

        ppg_data = np.array(normalized_windows)
        print(f"  Normalized shape: {ppg_data.shape}")
    else:
        ppg_data = np.array(windows)
        print(f"  All windows have same length: {window_lengths[0]}")

    # Load glucose labels
    print(f"Loading glucose labels from {glucose_file}...")
    glucose_df = pd.read_csv(glucose_file)
    glucose_df = glucose_df.sort_values('window_index')
    glucose_data = glucose_df['glucose_mg_dl'].values

    # Verify data alignment
    if len(ppg_data) != len(glucose_data):
        raise ValueError(f"Mismatch: {len(ppg_data)} PPG windows but {len(glucose_data)} glucose labels")

    # Check for NaN values
    if np.isnan(ppg_data).any():
        nan_count = np.isnan(ppg_data).sum()
        print(f"WARNING: Found {nan_count} NaN values in PPG data - removing affected windows")
        valid_mask = ~np.isnan(ppg_data).any(axis=1)
        ppg_data = ppg_data[valid_mask]
        glucose_data = glucose_data[valid_mask]

    if np.isnan(glucose_data).any():
        nan_count = np.isnan(glucose_data).sum()
        print(f"WARNING: Found {nan_count} NaN values in glucose data - removing affected windows")
        valid_mask = ~np.isnan(glucose_data)
        ppg_data = ppg_data[valid_mask]
        glucose_data = glucose_data[valid_mask]

    print(f"Loaded {len(ppg_data)} paired samples")
    print(f"  PPG shape: {ppg_data.shape}")
    print(f"  Glucose shape: {glucose_data.shape}")
    print(f"  PPG range: [{ppg_data.min():.4f}, {ppg_data.max():.4f}]")
    print(f"  Glucose range: [{glucose_data.min():.1f}, {glucose_data.max():.1f}] mg/dL")

    return ppg_data, glucose_data


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch

    Returns:
        avg_loss: average training loss
        avg_mae: average MAE in mg/dL
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for batch_idx, (ppg, glucose) in enumerate(train_loader):
        ppg = ppg.to(device)
        glucose = glucose.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(ppg)

        # Compute loss
        loss = criterion(predictions, glucose)

        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at batch {batch_idx}")
            print(f"  PPG stats: min={ppg.min():.4f}, max={ppg.max():.4f}, mean={ppg.mean():.4f}")
            print(f"  Glucose stats: min={glucose.min():.4f}, max={glucose.max():.4f}, mean={glucose.mean():.4f}")
            print(f"  Predictions stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()

        # Compute MAE (on normalized values)
        with torch.no_grad():
            mae = torch.abs(predictions - glucose).mean()
            total_mae += mae.item()

        num_batches += 1

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches

    return avg_loss, avg_mae


def validate(model, val_loader, criterion, device, dataset):
    """
    Validate the model

    Returns:
        avg_loss: average validation loss
        mae: MAE in mg/dL (denormalized)
        rmse: RMSE in mg/dL (denormalized)
        predictions: list of predictions
        actuals: list of actual values
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_actuals = []
    num_batches = 0

    with torch.no_grad():
        for ppg, glucose in val_loader:
            ppg = ppg.to(device)
            glucose = glucose.to(device)

            # Forward pass
            predictions = model(ppg)

            # Compute loss
            loss = criterion(predictions, glucose)
            total_loss += loss.item()
            num_batches += 1

            # Store predictions and actuals (normalized)
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_actuals.extend(glucose.cpu().numpy().flatten())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Denormalize to mg/dL
    predictions_mgdl = dataset.denormalize_glucose(all_predictions)
    actuals_mgdl = dataset.denormalize_glucose(all_actuals)

    # Compute metrics in mg/dL
    mae = mean_absolute_error(actuals_mgdl, predictions_mgdl)
    mse = mean_squared_error(actuals_mgdl, predictions_mgdl)
    rmse = np.sqrt(mse)

    avg_loss = total_loss / num_batches

    return avg_loss, mae, rmse, predictions_mgdl, actuals_mgdl


def plot_predictions(predictions, actuals, epoch, output_dir):
    """
    Plot predicted vs actual glucose values
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Glucose (mg/dL)')
    plt.ylabel('Predicted Glucose (mg/dL)')
    plt.title(f'Predicted vs Actual (Epoch {epoch})')
    plt.grid(True)

    # Residuals
    plt.subplot(1, 2, 2)
    residuals = predictions - actuals
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error (mg/dL)')
    plt.ylabel('Count')
    plt.title(f'Error Distribution (Epoch {epoch})')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_epoch_{epoch}.png'))
    plt.close()


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def train(args):
    """
    Main training function
    """
    print("=" * 80)
    print("ResNet34-1D Glucose Prediction - Training")
    print("=" * 80)
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'training_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Save training configuration
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Output directory: {output_dir}")
    print()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Load data
    print("Loading data...")
    ppg_data, glucose_data = load_data_from_csv(args.data_dir)
    print()

    # Create dataset
    dataset = PPGGlucoseDataset(ppg_data, glucose_data, normalize=True)

    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Data split:")
    print(f"  Training: {len(train_dataset)} samples ({args.train_split*100:.0f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({args.val_split*100:.0f}%)")
    print(f"  Test: {len(test_dataset)} samples ({(1-args.train_split-args.val_split)*100:.0f}%)")
    print()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    input_length = ppg_data.shape[1]
    model = ResNet34_1D(input_length=input_length, num_classes=1)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Architecture:")
    print(f"  Input length: {input_length} samples")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (float32)")
    print()

    # Loss function and optimizer
    criterion = nn.MSELoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = None

    print("Training Configuration:")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Scheduler: {args.scheduler}")
    print()

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Training loop
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()

    best_val_mae = float('inf')
    patience_counter = 0

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'learning_rate': []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss, val_mae, val_rmse, val_predictions, val_actuals = validate(
            model, val_loader, criterion, device, dataset
        )

        # Update learning rate
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_mae)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f} mg/dL")
        print(f"  Val RMSE: {val_rmse:.2f} mg/dL")
        print(f"  Learning Rate: {current_lr:.6f}")
        print()

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_mae'].append(val_mae)
        training_history['val_rmse'].append(val_rmse)
        training_history['learning_rate'].append(current_lr)

        # Check if best model
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            patience_counter = 0
            print(f"  [NEW BEST MODEL] MAE: {val_mae:.2f} mg/dL")
        else:
            patience_counter += 1

        # Save checkpoint
        if epoch % args.save_freq == 0 or is_best:
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse
            }
            save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best)

        # Plot predictions
        if epoch % args.plot_freq == 0:
            plot_predictions(val_predictions, val_actuals, epoch, output_dir)

        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best validation MAE: {best_val_mae:.2f} mg/dL")
            break

        print()

    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # Final evaluation on test set
    print("=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    # Load best model
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_mae, test_rmse, test_predictions, test_actuals = validate(
        model, test_loader, criterion, device, dataset
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f} mg/dL")
    print(f"Test RMSE: {test_rmse:.2f} mg/dL")
    print()

    # Plot final test results
    plot_predictions(test_predictions, test_actuals, 'test', output_dir)

    # Save final metrics
    final_metrics = {
        'best_val_mae': best_val_mae,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }

    with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Close TensorBoard writer
    writer.close()

    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model.pth')}")
    print(f"Training history saved to: {os.path.join(output_dir, 'training_history.csv')}")
    print(f"TensorBoard logs: {os.path.join(output_dir, 'tensorboard')}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Train ResNet34-1D for Glucose Prediction')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing ppg_windows.csv and glucose_labels.csv')
    parser.add_argument('--output_dir', type=str, default='./training_outputs',
                        help='Directory to save training outputs')

    # Data split
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Fraction of data for training (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data for validation (default: 0.15)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer (default: adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization (default: 1e-4)')

    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler (default: step)')
    parser.add_argument('--scheduler_step', type=int, default=30,
                        help='Step size for StepLR scheduler (default: 30)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler (default: 0.1)')

    # Checkpoint and logging
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--plot_freq', type=int, default=10,
                        help='Plot predictions every N epochs (default: 10)')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience (0 to disable, default: 20)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Train
    train(args)


if __name__ == '__main__':
    main()
