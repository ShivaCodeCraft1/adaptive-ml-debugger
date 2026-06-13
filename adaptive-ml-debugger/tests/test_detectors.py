import math
import pytest
from anomaly_detection.exploding_gradients import detect_exploding_gradients
from anomaly_detection.vanishing_gradients import detect_vanishing_gradients
from anomaly_detection.divergence import detect_divergence
from anomaly_detection.overfitting import detect_overfitting

def test_exploding_gradients_detected():
    # Test exceeding threshold
    grad_norms = [1.0, 5.0, 15.0]
    result = detect_exploding_gradients(grad_norms, threshold=10.0)
    assert result["detected"] is True
    assert "exceeds threshold" in result["message"]

    # Test NaN detection
    grad_norms_nan = [1.0, float('nan')]
    result_nan = detect_exploding_gradients(grad_norms_nan)
    assert result_nan["detected"] is True
    assert "NaN or Infinity" in result_nan["message"]

def test_exploding_gradients_stable():
    grad_norms = [1.0, 2.0, 3.0, 4.0]
    result = detect_exploding_gradients(grad_norms, threshold=10.0)
    assert result["detected"] is False

def test_vanishing_gradients_detected():
    # Window is 3, all below 1e-4
    grad_norms = [1e-5, 1e-6, 1e-7]
    result = detect_vanishing_gradients(grad_norms, threshold=1e-4, window=3)
    assert result["detected"] is True

def test_vanishing_gradients_stable():
    # One value is above threshold, breaking the consecutive window
    grad_norms = [1e-5, 1.0, 1e-5]
    result = detect_vanishing_gradients(grad_norms, threshold=1e-4, window=3)
    assert result["detected"] is False

def test_divergence_detected():
    train_losses = [1.0, 0.8, 0.5]
    val_losses = [1.2, 1.5, 2.0]
    # Ratio of val (2.0) to train (0.5) is 4.0, which > 3.0 threshold
    result = detect_divergence(train_losses, val_losses, ratio_threshold=3.0)
    assert result["detected"] is True

def test_divergence_stable():
    train_losses = [1.0, 0.8, 0.5]
    val_losses = [1.2, 1.0, 0.6]
    # Ratio is 1.2, which is <= 3.0
    result = detect_divergence(train_losses, val_losses, ratio_threshold=3.0)
    assert result["detected"] is False

def test_overfitting_detected():
    # Train decreases, val strictly increases for patience=3 epochs (4 data points needed)
    train_losses = [1.0, 0.9, 0.8, 0.7]
    val_losses = [1.0, 1.1, 1.2, 1.3]
    result = detect_overfitting(train_losses, val_losses, patience=3)
    assert result["detected"] is True

def test_overfitting_stable():
    # Both decrease
    train_losses = [1.0, 0.9, 0.8, 0.7]
    val_losses = [1.2, 1.1, 1.0, 0.9]
    result = detect_overfitting(train_losses, val_losses, patience=3)
    assert result["detected"] is False