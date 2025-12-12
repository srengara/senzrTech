"""
ResNet34-1D for Glucose Prediction from PPG Signals

This module implements a 1D ResNet34 architecture with residual connections
for predicting glucose levels from filtered PPG signal windows.

Architecture:
- Input: PPG signal windows (batch_size, 1, sequence_length)
- Residual blocks with 1D convolutions
- Output: Glucose value (mg/dL)

Based on the ResNet architecture adapted for 1D time-series data.

Integration with VitalDB PPG Pipeline:
--------------------------------------
This model takes the output from peak_detection.py's
ppg_peak_detection_pipeline_with_template() function:
- Input: filtered_windows (List[np.ndarray])
- Output: Predicted glucose values (mg/dL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
import os


class ResidualBlock1D(nn.Module):
    """
    Basic Residual Block for 1D Convolutions.

    Structure:
    - Conv1D -> BatchNorm -> ReLU -> Conv1D -> BatchNorm
    - Skip connection with optional downsampling
    - Final ReLU activation

    This follows the ResNet design where the skip connection allows
    gradients to flow directly through the network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize Residual Block.

        Parameters:
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Convolution kernel size (default: 3)
        stride : int
            Stride for first convolution (default: 1)
        downsample : nn.Module, optional
            Downsampling layer for skip connection if dimensions change
        """
        super(ResidualBlock1D, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second convolution layer
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Downsample layer for skip connection
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
        -------
        out : torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length')
        """
        # Store input for skip connection
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling to skip connection if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection (residual connection)
        out += identity
        out = self.relu(out)

        return out


class ResNet34_1D(nn.Module):
    """
    ResNet34 architecture adapted for 1D PPG signals with glucose prediction output.

    Architecture Overview:
    ---------------------
    - Initial Conv layer: 1 channel -> 64 channels
    - Layer 1: 3 residual blocks (64 channels)
    - Layer 2: 4 residual blocks (128 channels, stride 2)
    - Layer 3: 6 residual blocks (256 channels, stride 2)
    - Layer 4: 3 residual blocks (512 channels, stride 2)
    - Adaptive Average Pooling
    - Fully Connected Layer -> Glucose value

    Total: 3 + 4 + 6 + 3 = 16 residual blocks (32 conv layers)
    Plus initial conv and FC = 34 layers total
    """

    def __init__(
        self,
        input_length: int = 100,
        num_classes: int = 1,
        dropout_rate: float = 0.5
    ):
        """
        Initialize ResNet34-1D for glucose prediction.

        Parameters:
        ----------
        input_length : int
            Expected length of input PPG window (default: 100 samples)
        num_classes : int
            Number of output values (default: 1 for glucose prediction)
        dropout_rate : float
            Dropout probability for regularization (default: 0.5)
        """
        super(ResNet34_1D, self).__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        # Initial convolution layer
        self.conv1 = nn.Conv1d(
            in_channels=1,  # Single channel PPG signal
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (following ResNet34 architecture)
        # Layer 1: 3 blocks, 64 channels
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)

        # Layer 2: 4 blocks, 128 channels
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)

        # Layer 3: 6 blocks, 256 channels
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)

        # Layer 4: 3 blocks, 512 channels
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer for glucose prediction
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a layer consisting of multiple residual blocks.

        Parameters:
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        num_blocks : int
            Number of residual blocks in this layer
        stride : int
            Stride for first block (for downsampling)

        Returns:
        -------
        layer : nn.Sequential
            Sequential container of residual blocks
        """
        downsample = None

        # Create downsampling layer if channels or spatial dims change
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

        layers = []

        # First block (may have downsampling)
        layers.append(
            ResidualBlock1D(
                in_channels,
                out_channels,
                stride=stride,
                downsample=downsample
            )
        )

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock1D(out_channels, out_channels)
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet34-1D.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, sequence_length)
            PPG signal windows

        Returns:
        -------
        glucose : torch.Tensor
            Predicted glucose values of shape (batch_size, 1)
            Values in mg/dL
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers (residual connections)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Dropout for regularization
        x = self.dropout(x)

        # Fully connected layer for glucose prediction
        glucose = self.fc(x)

        return glucose


class GlucosePredictor:
    """
    High-level wrapper for ResNet34-1D glucose prediction.

    Provides easy-to-use interface for:
    - Model initialization
    - Training
    - Inference
    - Model saving/loading
    - Data preprocessing

    Usage Example:
    --------------
    from src.training.resnet34_glucose_predictor import GlucosePredictor

    # Initialize predictor
    predictor = GlucosePredictor(input_length=100)

    # Predict from filtered windows (from peak_detection.py)
    glucose_values = predictor.predict(filtered_windows)
    """

    def __init__(
        self,
        input_length: int = 100,
        device: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize Glucose Predictor.

        Parameters:
        ----------
        input_length : int
            Expected length of PPG windows (default: 100 samples)
            This should match the window size from peak detection
        device : str, optional
            Device to run model on ('cpu' or 'cuda')
            If None, automatically selects CUDA if available
        model_path : str, optional
            Path to pre-trained model weights
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = ResNet34_1D(input_length=input_length).to(self.device)

        # Load pre-trained weights if provided
        if model_path is not None and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded model from {model_path}")

        self.input_length = input_length

    def preprocess_windows(
        self,
        windows: List[np.ndarray],
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess PPG windows for model input.

        Steps:
        1. Normalize each window to zero mean and unit variance
        2. Pad or truncate to target length
        3. Convert to PyTorch tensor
        4. Add channel dimension

        Parameters:
        ----------
        windows : List[np.ndarray]
            List of PPG signal windows from peak detection
        target_length : int, optional
            Target length for padding/truncation (default: self.input_length)

        Returns:
        -------
        tensor : torch.Tensor
            Preprocessed tensor of shape (batch_size, 1, target_length)
        """
        if target_length is None:
            target_length = self.input_length

        processed_windows = []

        for window in windows:
            # Normalize to zero mean and unit variance
            window_mean = np.mean(window)
            window_std = np.std(window)
            window_normalized = (window - window_mean) / (window_std + 1e-8)

            # Pad or truncate to target length
            if len(window_normalized) < target_length:
                # Pad with zeros
                padding = target_length - len(window_normalized)
                window_padded = np.pad(
                    window_normalized,
                    (0, padding),
                    mode='constant',
                    constant_values=0
                )
            else:
                # Truncate
                window_padded = window_normalized[:target_length]

            processed_windows.append(window_padded)

        # Convert to numpy array
        windows_array = np.array(processed_windows)

        # Add channel dimension: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        windows_array = np.expand_dims(windows_array, axis=1)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(windows_array).float()

        return tensor

    def predict(
        self,
        windows: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict glucose values from PPG windows.

        Parameters:
        ----------
        windows : List[np.ndarray]
            List of filtered PPG windows from peak detection pipeline
        batch_size : int
            Batch size for inference (default: 32)

        Returns:
        -------
        glucose_values : np.ndarray
            Predicted glucose values in mg/dL, shape (num_windows,)
        """
        self.model.eval()

        # Preprocess windows
        windows_tensor = self.preprocess_windows(windows)

        # Split into batches
        num_windows = len(windows_tensor)
        glucose_predictions = []

        with torch.no_grad():
            for i in range(0, num_windows, batch_size):
                batch = windows_tensor[i:i + batch_size].to(self.device)

                # Forward pass through ResNet34
                glucose_batch = self.model(batch)

                # Move to CPU and convert to numpy
                glucose_predictions.append(glucose_batch.cpu().numpy())

        # Concatenate all predictions
        glucose_values = np.concatenate(glucose_predictions, axis=0).squeeze()

        return glucose_values

    def predict_single(self, window: np.ndarray) -> float:
        """
        Predict glucose value from a single PPG window.

        Parameters:
        ----------
        window : np.ndarray
            Single PPG window

        Returns:
        -------
        glucose : float
            Predicted glucose value in mg/dL
        """
        glucose_values = self.predict([window])
        return float(glucose_values[0])

    def predict_with_stats(
        self,
        windows: List[np.ndarray],
        batch_size: int = 32
    ) -> Dict[str, any]:
        """
        Predict glucose values with statistical summary.

        Parameters:
        ----------
        windows : List[np.ndarray]
            List of filtered PPG windows
        batch_size : int
            Batch size for inference

        Returns:
        -------
        results : Dict
            Dictionary containing:
            - 'predictions': Individual glucose predictions
            - 'mean_glucose': Mean glucose value
            - 'std_glucose': Standard deviation
            - 'min_glucose': Minimum value
            - 'max_glucose': Maximum value
            - 'num_windows': Number of windows processed
        """
        predictions = self.predict(windows, batch_size)

        return {
            'predictions': predictions,
            'mean_glucose': float(np.mean(predictions)),
            'std_glucose': float(np.std(predictions)),
            'min_glucose': float(np.min(predictions)),
            'max_glucose': float(np.max(predictions)),
            'num_windows': len(predictions)
        }

    def train_step(
        self,
        windows: List[np.ndarray],
        glucose_targets: np.ndarray,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Perform single training step.

        Parameters:
        ----------
        windows : List[np.ndarray]
            Batch of PPG windows
        glucose_targets : np.ndarray
            Target glucose values (mg/dL)
        optimizer : torch.optim.Optimizer
            Optimizer instance (e.g., Adam)
        criterion : nn.Module
            Loss function (e.g., nn.MSELoss())

        Returns:
        -------
        loss : float
            Training loss value
        """
        self.model.train()

        # Preprocess windows
        windows_tensor = self.preprocess_windows(windows).to(self.device)
        targets_tensor = torch.from_numpy(glucose_targets).float().to(self.device)

        # Ensure targets have correct shape
        if len(targets_tensor.shape) == 1:
            targets_tensor = targets_tensor.unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = self.model(windows_tensor)

        # Compute loss
        loss = criterion(predictions, targets_tensor)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        return loss.item()

    def evaluate(
        self,
        windows: List[np.ndarray],
        glucose_targets: np.ndarray,
        criterion: nn.Module,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.

        Parameters:
        ----------
        windows : List[np.ndarray]
            PPG windows
        glucose_targets : np.ndarray
            True glucose values (mg/dL)
        criterion : nn.Module
            Loss function
        batch_size : int
            Batch size for evaluation

        Returns:
        -------
        metrics : Dict[str, float]
            Dictionary containing:
            - 'loss': Average loss
            - 'mae': Mean Absolute Error (mg/dL)
            - 'rmse': Root Mean Squared Error (mg/dL)
        """
        self.model.eval()

        # Get predictions
        predictions = self.predict(windows, batch_size=batch_size)

        # Compute metrics
        mae = np.mean(np.abs(predictions - glucose_targets))
        rmse = np.sqrt(np.mean((predictions - glucose_targets) ** 2))

        # Compute loss
        windows_tensor = self.preprocess_windows(windows).to(self.device)
        targets_tensor = torch.from_numpy(glucose_targets).float().to(self.device)
        if len(targets_tensor.shape) == 1:
            targets_tensor = targets_tensor.unsqueeze(1)

        with torch.no_grad():
            pred_tensor = self.model(windows_tensor)
            loss = criterion(pred_tensor, targets_tensor).item()

        return {
            'loss': loss,
            'mae': mae,
            'rmse': rmse
        }

    def save_model(self, save_path: str):
        """
        Save model weights.

        Parameters:
        ----------
        save_path : str
            Path to save model weights (.pth file)
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_length': self.input_length
        }, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load model weights.

        Parameters:
        ----------
        load_path : str
            Path to load model weights from (.pth file)
        """
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Handle legacy checkpoints that don't have input_length
        if 'input_length' in checkpoint:
            self.input_length = checkpoint['input_length']
        print(f"Model loaded from {load_path}")

    def get_model_summary(self) -> str:
        """
        Get model architecture summary.

        Returns:
        -------
        summary : str
            String representation of model architecture
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary = f"""
ResNet34-1D Glucose Predictor
==============================
Input Length: {self.input_length} samples
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Device: {self.device}

Architecture:
- Initial Conv1D: 1 -> 64 channels (kernel=7, stride=2)
- MaxPool1D (kernel=3, stride=2)
- Layer 1: 3 residual blocks (64 channels)
- Layer 2: 4 residual blocks (128 channels, stride=2)
- Layer 3: 6 residual blocks (256 channels, stride=2)
- Layer 4: 3 residual blocks (512 channels, stride=2)
- Adaptive Avg Pool + Dropout(0.5) + FC -> Glucose value (mg/dL)

Total ResNet Layers: 34
"""
        return summary
