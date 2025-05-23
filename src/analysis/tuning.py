import numpy as np
from scipy.optimize import curve_fit
from utils import circular_distance_matrix


def gaussian_smooth(data, sigma=6):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(data, sigma, mode='wrap')


def vonmises(x, mu, kappa, A, B):
    return A * np.exp(kappa * np.cos(x-mu)) + B


def compute_tuning(spikes, theta_bins, occupancy, sigma=6):
    rate = spikes / occupancy
    rate_smooth = gaussian_smooth(rate, sigma)
    return rate_smooth


def fit_vonmises(rate, theta):
    p0 = [theta[np.argmax(rate)], 1.0, np.max(rate)-np.min(rate), np.min(rate)]
    params, _ = curve_fit(lambda x, mu, kappa, A, B: vonmises(x, mu, kappa, A, B), theta, rate, p0)
    return params