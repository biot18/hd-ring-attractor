import numpy as np


def compute_drift_speed(decoded_angles, dt):
    diffs = np.diff(decoded_angles)
    diffs = (diffs + np.pi) % (2*np.pi) - np.pi
    return np.mean(np.abs(diffs)) / dt


def compute_noise_correlation(rates):
    # rates: shape (T, N)
    rates_centered = rates - rates.mean(axis=0)
    cov = np.cov(rates_centered, rowvar=False)
    std = rates_centered.std(axis=0)
    corr = cov / np.outer(std, std)
    return corr