import numpy as np

def bayesian_decode(spikes, tuning_curves):
    """
    spikes: shape (N,)
    tuning_curves: shape (N, M)  # M: bin count over circle
    returns posterior over M bins
    """
    log_likelihood = spikes[:, None] * np.log(tuning_curves + 1e-12) - tuning_curves
    posterior = np.exp(log_likelihood.sum(axis=0))
    return posterior / posterior.sum()