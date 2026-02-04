import numpy as np


def threshold_fixed(magnitude, threshold):
    """Fixed threshold. Returns uint8 0-255"""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    mask = magnitude >= threshold
    return (mask.astype(np.uint8) * 255)


def threshold_percentile(magnitude, percentile=90.0):
    """Percentile threshold. Returns uint8 (0-255) and threshold value"""
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be in [0, 100]")
    thr = float(np.percentile(magnitude, percentile))
    mask = magnitude >= thr
    return (mask.astype(np.uint8) * 255), thr
