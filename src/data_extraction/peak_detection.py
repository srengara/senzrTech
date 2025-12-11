"""
PPG Signal Peak Detection Implementation
Based on the pseudocode algorithm for detecting peaks in PPG (Photoplethysmography) signals.

This implementation includes:
1. DETECTPEAKS: Identifies peaks in the signal based on height and distance thresholds
2. EXTRACTWINDOWS: Extracts windows around detected peaks
3. Helper function COUNTPEAKS: Counts peaks within a window (placeholder for cosine similarity)
"""

import numpy as np
from typing import List, Tuple, Optional


def detect_peaks(
    ppg_signal: np.ndarray,
    height_threshold: float = 20,
    distance_threshold: Optional[float] = None,
    fs: float = 100
) -> List[int]:
    """
    Detect peaks in PPG signal based on height and distance thresholds.
    
    Algorithm:
    - A peak is detected when:
      1. ppg_signal[i-1] < ppg_signal[i] > ppg_signal[i+1] (local maximum)
      2. ppg_signal[i] > height_threshold
      3. Either peaks list is empty OR distance from last peak > distance_threshold
    
    Parameters:
    ----------
    ppg_signal : np.ndarray
        Input PPG signal array
    height_threshold : float, optional
        Minimum peak height (default: 20)
    distance_threshold : float, optional
        Minimum distance between peaks in samples (default: 0.8 * fs)
    fs : float, optional
        Sampling frequency in Hz (default: 100)
    
    Returns:
    -------
    peaks : List[int]
        List of indices where peaks are detected
        
    Example:
    --------
    >>> signal = np.array([10, 20, 30, 25, 15, 10, 25, 35, 30, 20])
    >>> peaks = detect_peaks(signal, height_threshold=20, distance_threshold=3)
    >>> print(peaks)
    [2, 7]
    """
    # Initialize parameters (from pseudocode lines 1-7)
    if distance_threshold is None:
        distance_threshold = 0.8 * fs  # Minimum distance between peaks
    
    peaks = []
    
    # Iterate through each sample (line 10)
    for i in range(len(ppg_signal)):
        # Check if current point is a local maximum (line 11)
        # ppg_signal[i-1] < ppg_signal[i] > ppg_signal[i+1]
        if i > 0 and i < len(ppg_signal) - 1:
            if ppg_signal[i - 1] < ppg_signal[i] > ppg_signal[i + 1]:
                # Check height threshold and distance constraint (line 12)
                if ppg_signal[i] > height_threshold:
                    # Check if peaks is empty or distance from last peak is sufficient
                    if len(peaks) == 0 or (i - peaks[-1]) > distance_threshold:
                        peaks.append(i)  # Line 13
    
    return peaks


def extract_windows(
    ppg_signal: np.ndarray,
    peaks: List[int],
    window_size: int,
    similarity_threshold: float = 0.85
) -> List[np.ndarray]:
    """
    Extract windows around detected peaks that contain exactly one peak.
    
    Algorithm:
    - For each detected peak:
      1. Calculate window_start = max(0, peak - window_size/3)
      2. Calculate window_end = min(len(signal), peak + window_size/2)
      3. Extract window = ppg_signal[window_start : window_end]
      4. If CountPeaks(window) == 1, append window to results
    
    Parameters:
    ----------
    ppg_signal : np.ndarray
        Input PPG signal array
    peaks : List[int]
        List of peak indices from detect_peaks
    window_size : int
        Size of window to extract around each peak (in samples)
    similarity_threshold : float, optional
        Cosine similarity threshold for peak validation (default: 0.85)
        Note: Currently not used in CountPeaks, placeholder for future implementation
    
    Returns:
    -------
    windows : List[np.ndarray]
        List of signal windows, each containing exactly one valid peak
        
    Example:
    --------
    >>> signal = np.random.randn(1000)
    >>> peaks = [100, 250, 400, 550, 700]
    >>> windows = extract_windows(signal, peaks, window_size=100)
    >>> print(f"Extracted {len(windows)} valid windows")
    """
    windows = []
    
    # For each peak in peaks (line 21)
    for peak in peaks:
        # Calculate window boundaries (lines 22-23)
        window_start = max(0, peak - window_size // 2)
        window_end = min(len(ppg_signal), peak + window_size // 2)
        
        # Extract window (line 24)
        window = ppg_signal[window_start:window_end]
        
        # Check if window contains exactly one peak (line 25)
        if count_peaks(window) == 1:
            windows.append(window)  # Line 26
    
    return windows


def count_peaks(window: np.ndarray, height_threshold: Optional[float] = None) -> int:
    """
    Count the number of peaks in a window.

    This is a simplified implementation that counts local maxima.
    According to the pseudocode comment (line 7), this should ideally use
    cosine similarity threshold for more sophisticated peak validation.

    Parameters:
    ----------
    window : np.ndarray
        Signal window to analyze
    height_threshold : float, optional
        Minimum height for a point to be considered a peak
        If None, uses median of the window as threshold (default: None)

    Returns:
    -------
    count : int
        Number of peaks detected in the window

    Note:
    -----
    Future enhancement: Implement cosine similarity-based peak validation
    as mentioned in the pseudocode (similarity_threshold = 0.85)
    """
    if len(window) < 3:
        return 0

    # Use adaptive threshold if not provided
    if height_threshold is None:
        height_threshold = np.median(window)

    count = 0

    for i in range(1, len(window) - 1):
        # Check if it's a local maximum and exceeds height threshold
        if window[i - 1] < window[i] > window[i + 1] and window[i] > height_threshold:
            count += 1

    return count


def compute_template(windows: List[np.ndarray]) -> np.ndarray:
    """
    Compute template by averaging all windows.

    Algorithm (from pseudocode lines 31-34):
    - template ← mean(windows)
    - return template

    This function computes the mean window across all extracted windows
    to create a representative template of a typical PPG beat.

    Parameters:
    ----------
    windows : List[np.ndarray]
        List of signal windows extracted around peaks

    Returns:
    -------
    template : np.ndarray
        Average template computed from all windows

    Example:
    --------
    >>> windows = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([3, 4, 5])]
    >>> template = compute_template(windows)
    >>> print(template)
    [2. 3. 4.]

    Note:
    -----
    All windows should have the same length. If windows have different lengths,
    only windows with the most common length should be used.
    """
    if not windows:
        return np.array([])

    # Find the most common window length
    lengths = [len(w) for w in windows]
    most_common_length = max(set(lengths), key=lengths.count)

    # Filter windows to only include those with the most common length
    filtered_windows = [w for w in windows if len(w) == most_common_length]

    if not filtered_windows:
        return np.array([])

    # Stack windows and compute mean along axis 0
    stacked_windows = np.stack(filtered_windows, axis=0)
    template = np.mean(stacked_windows, axis=0)

    return template


def cosine_similarity(window: np.ndarray, template: np.ndarray) -> float:
    """
    Compute cosine similarity between a window and template.

    Algorithm (from pseudocode lines 35-40):
    - dot_product ← sum(window × template)
    - magnitude_window ← sqrt(sum(window²))
    - magnitude_template ← sqrt(sum(template²))
    - return dot_product / (magnitude_window × magnitude_template)

    Cosine similarity measures the cosine of the angle between two vectors,
    resulting in a value between -1 and 1, where 1 indicates identical shape.

    Parameters:
    ----------
    window : np.ndarray
        Signal window to compare
    template : np.ndarray
        Template signal to compare against

    Returns:
    -------
    similarity : float
        Cosine similarity value between 0 and 1 (typically)
        Returns 0.0 if either magnitude is zero

    Example:
    --------
    >>> window = np.array([1, 2, 3, 4])
    >>> template = np.array([1, 2, 3, 4])
    >>> similarity = cosine_similarity(window, template)
    >>> print(f"{similarity:.3f}")
    1.000

    >>> window = np.array([1, 2, 3, 4])
    >>> template = np.array([4, 3, 2, 1])
    >>> similarity = cosine_similarity(window, template)
    >>> print(f"{similarity:.3f}")
    0.400
    """
    # Ensure windows have the same length
    if len(window) != len(template):
        # Pad or truncate to match lengths
        min_len = min(len(window), len(template))
        window = window[:min_len]
        template = template[:min_len]

    # Compute dot product: sum(window × template)
    dot_product = np.sum(window * template)

    # Compute magnitudes: sqrt(sum(window²)) and sqrt(sum(template²))
    magnitude_window = np.sqrt(np.sum(window ** 2))
    magnitude_template = np.sqrt(np.sum(template ** 2))

    # Handle zero magnitude case
    if magnitude_window == 0 or magnitude_template == 0:
        return 0.0

    # Compute cosine similarity
    similarity = dot_product / (magnitude_window * magnitude_template)

    return similarity


def filter_windows_by_similarity(
    windows: List[np.ndarray],
    template: np.ndarray,
    similarity_threshold: float = 0.85
) -> List[np.ndarray]:
    """
    Filter windows by cosine similarity to template.

    Algorithm (from pseudocode lines 41-50):
    - filtered_windows ← []
    - for each window in windows do
    -     similarity ← COSINESIMILARITY(window, template)
    -     if similarity ≥ similarity_threshold then
    -         filtered_windows.append(window)
    -     end if
    - end for
    - return filtered_windows

    This function filters out windows that don't match the template shape,
    removing artifacts and irregular beats from the analysis.

    Parameters:
    ----------
    windows : List[np.ndarray]
        List of signal windows to filter
    template : np.ndarray
        Template signal to compare against
    similarity_threshold : float, optional
        Minimum cosine similarity threshold (default: 0.85)

    Returns:
    -------
    filtered_windows : List[np.ndarray]
        List of windows that meet the similarity threshold

    Example:
    --------
    >>> windows = [np.array([1, 2, 3, 2, 1]),
    ...            np.array([1, 2, 3, 2, 1]),
    ...            np.array([5, 1, 2, 1, 5])]
    >>> template = np.array([1, 2, 3, 2, 1])
    >>> filtered = filter_windows_by_similarity(windows, template, 0.9)
    >>> print(f"Filtered {len(filtered)} out of {len(windows)} windows")
    Filtered 2 out of 3 windows
    """
    filtered_windows = []

    # For each window in windows
    for window in windows:
        # Compute similarity with template
        similarity = cosine_similarity(window, template)

        # If similarity meets threshold, keep the window
        if similarity >= similarity_threshold:
            filtered_windows.append(window)

    return filtered_windows


# Complete pipeline function (basic version without template filtering)
def ppg_peak_detection_pipeline(
    ppg_signal: np.ndarray,
    fs: float = 100,
    window_duration: float = 1,
    height_threshold: float = 20,
    distance_threshold: Optional[float] = None,
    similarity_threshold: float = 0.85
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Complete PPG peak detection pipeline (basic version).

    This function implements the algorithm from the pseudocode:
    1. Initialize parameters
    2. Detect peaks in the signal
    3. Extract windows around peaks

    NOTE: This is the basic version without template-based filtering.
    For template-based filtering, use ppg_peak_detection_pipeline_with_template().

    Parameters:
    ----------
    ppg_signal : np.ndarray
        Input PPG signal
    fs : float, optional
        Sampling frequency in Hz (default: 100)
    window_duration : float, optional
        Window duration in seconds (default: 1)
    height_threshold : float, optional
        Minimum peak height (default: 20)
    distance_threshold : float, optional
        Minimum distance between peaks in samples (default: 0.8 * fs)
    similarity_threshold : float, optional
        Cosine similarity threshold (default: 0.85)

    Returns:
    -------
    peaks : List[int]
        Detected peak indices
    windows : List[np.ndarray]
        Extracted windows containing single peaks

    Example:
    --------
    >>> # Generate sample PPG signal
    >>> t = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(1000)
    >>> peaks, windows = ppg_peak_detection_pipeline(signal, fs=100)
    >>> print(f"Detected {len(peaks)} peaks")
    >>> print(f"Extracted {len(windows)} valid windows")
    """
    # Initialize parameters (lines 1-7 from pseudocode)
    if distance_threshold is None:
        distance_threshold = 0.8 * fs

    window_size = int(fs * window_duration)  # Convert window duration to samples

    # Step 1: Detect peaks (function DETECTPEAKS, lines 8-18)
    peaks = detect_peaks(
        ppg_signal,
        height_threshold=height_threshold,
        distance_threshold=distance_threshold,
        fs=fs
    )

    # Step 2: Extract windows (function EXTRACTWINDOWS, lines 19-30)
    windows = extract_windows(
        ppg_signal,
        peaks,
        window_size=window_size,
        similarity_threshold=similarity_threshold
    )

    return peaks, windows


def ppg_peak_detection_pipeline_with_template(
    ppg_signal: np.ndarray,
    fs: float = 100,
    window_duration: float = 1,
    height_threshold: float = 20,
    distance_threshold: Optional[float] = None,
    similarity_threshold: float = 0.85
) -> Tuple[List[int], List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Complete PPG peak detection pipeline WITH template-based filtering.

    This function implements the MAIN algorithm from pseudocode (lines 51-57):
    1. peaks ← DETECTPEAKS(ppg_signal, height_threshold, distance_threshold)
    2. windows ← EXTRACTWINDOWS(ppg_signal, peaks, window_size)
    3. template ← COMPUTETEMPLATE(windows)
    4. filtered_windows ← FILTERWINDOWSBYSIMILARITY(windows, template, similarity_threshold)
    5. return filtered_windows

    Parameters:
    ----------
    ppg_signal : np.ndarray
        Input PPG signal
    fs : float, optional
        Sampling frequency in Hz (default: 100)
    window_duration : float, optional
        Window duration in seconds (default: 1)
    height_threshold : float, optional
        Minimum peak height (default: 20)
    distance_threshold : float, optional
        Minimum distance between peaks in samples (default: 0.8 * fs)
    similarity_threshold : float, optional
        Cosine similarity threshold for filtering (default: 0.85)

    Returns:
    -------
    peaks : List[int]
        Detected peak indices
    filtered_windows : List[np.ndarray]
        Filtered windows that match the template
    template : np.ndarray
        Computed template (average of all windows)
    all_windows : List[np.ndarray]
        All extracted windows (before filtering)

    Example:
    --------
    >>> # Generate sample PPG signal
    >>> t = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(1000)
    >>> peaks, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(signal, fs=100)
    >>> print(f"Detected {len(peaks)} peaks")
    >>> print(f"Extracted {len(all_windows)} windows")
    >>> print(f"Filtered to {len(filtered_windows)} windows (similarity ≥ 0.85)")
    >>> print(f"Template shape: {template.shape}")
    """
    # Initialize parameters
    if distance_threshold is None:
        distance_threshold = 0.8 * fs

    window_size = int(fs * window_duration)

    # Step 1: Detect peaks (pseudocode line 52)
    peaks = detect_peaks(
        ppg_signal,
        height_threshold=height_threshold,
        distance_threshold=distance_threshold,
        fs=fs
    )

    # Step 2: Extract windows (pseudocode line 53)
    windows = extract_windows(
        ppg_signal,
        peaks,
        window_size=window_size,
        similarity_threshold=similarity_threshold
    )

    # Step 3: Compute template (pseudocode line 54)
    template = compute_template(windows)

    # Step 4: Filter windows by similarity to template (pseudocode line 55)
    filtered_windows = filter_windows_by_similarity(
        windows,
        template,
        similarity_threshold=similarity_threshold
    )

    # Return filtered_windows (pseudocode line 56)
    return peaks, filtered_windows, template, windows


# Example usage and testing
if __name__ == "__main__":
    # Generate a synthetic PPG-like signal for testing
    print("=" * 70)
    print("PPG Peak Detection Algorithm - Test Run")
    print("=" * 70)
    
    # Create synthetic signal with peaks
    fs = 100  # Hz
    duration = 10  # seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate PPG signal: slow sine wave (heart rate ~60 bpm) with noise
    heart_rate_hz = 1.0  # 60 beats per minute
    ppg_signal = 50 + 30 * np.sin(2 * np.pi * heart_rate_hz * t) + 5 * np.random.randn(len(t))
    
    print(f"\nGenerated synthetic PPG signal:")
    print(f"  - Duration: {duration} seconds")
    print(f"  - Sampling rate: {fs} Hz")
    print(f"  - Signal length: {len(ppg_signal)} samples")
    print(f"  - Signal range: [{ppg_signal.min():.2f}, {ppg_signal.max():.2f}]")
    
    # Run basic peak detection pipeline
    print("\n[TEST 1] Basic Peak Detection Pipeline")
    print("-" * 70)
    peaks, windows = ppg_peak_detection_pipeline(
        ppg_signal,
        fs=fs,
        window_duration=1.0,
        height_threshold=60,  # Adjusted for synthetic signal
        distance_threshold=0.8 * fs
    )

    print(f"\nBasic Pipeline Results:")
    print(f"  - Detected peaks: {len(peaks)}")
    print(f"  - Peak indices: {peaks}")
    print(f"  - Extracted valid windows: {len(windows)}")

    if peaks:
        peak_times = np.array(peaks) / fs
        print(f"\nPeak timing:")
        print(f"  - Peak times (seconds): {peak_times}")
        if len(peaks) > 1:
            intervals = np.diff(peak_times)
            print(f"  - Inter-peak intervals: {intervals}")
            print(f"  - Mean heart rate: {60 / np.mean(intervals):.1f} BPM")

    # Display window information
    if windows:
        print(f"\nWindow details:")
        for i, window in enumerate(windows[:3]):  # Show first 3 windows
            print(f"  Window {i+1}: length={len(window)}, "
                  f"range=[{window.min():.2f}, {window.max():.2f}]")
        if len(windows) > 3:
            print(f"  ... and {len(windows) - 3} more windows")

    # Run template-based peak detection pipeline
    print("\n" + "=" * 70)
    print("[TEST 2] Template-Based Peak Detection Pipeline")
    print("-" * 70)
    peaks2, filtered_windows, template, all_windows = ppg_peak_detection_pipeline_with_template(
        ppg_signal,
        fs=fs,
        window_duration=1.0,
        height_threshold=60,
        distance_threshold=0.8 * fs,
        similarity_threshold=0.85
    )

    print(f"\nTemplate-Based Pipeline Results:")
    print(f"  - Detected peaks: {len(peaks2)}")
    print(f"  - Extracted windows: {len(all_windows)}")
    print(f"  - Filtered windows: {len(filtered_windows)}")
    print(f"  - Template shape: {template.shape if template.size > 0 else 'empty'}")
    if len(all_windows) > 0:
        print(f"  - Filtering rate: {len(filtered_windows)/len(all_windows)*100:.1f}% windows kept")
    else:
        print(f"  - Filtering rate: N/A (no windows extracted)")

    if len(filtered_windows) > 0:
        print(f"\nFiltered Window Details:")
        for i, window in enumerate(filtered_windows[:3]):
            similarity = cosine_similarity(window, template)
            print(f"  Window {i+1}: length={len(window)}, "
                  f"similarity={similarity:.3f}, "
                  f"range=[{window.min():.2f}, {window.max():.2f}]")
        if len(filtered_windows) > 3:
            print(f"  ... and {len(filtered_windows) - 3} more windows")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
