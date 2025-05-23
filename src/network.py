import numpy as np
from config import N, kappa_exc, A_exc, B_inh

def circular_distance(theta_i, theta_j):
    """環状距離を計算"""
    d = np.abs(theta_i - theta_j)
    return np.minimum(d, 2*np.pi - d)


def build_vonmises_weights(N=N, kappa=kappa_exc, A=A_exc, B=B_inh):
    """
    von Mises 型 Mexican-hat 結合行列を生成
    Returns
    -------
    W : ndarray, shape (N, N)
    """
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = circular_distance(theta[i], theta[j])
            W[i, j] = A * np.exp(kappa * np.cos(d)) - B
    # 行和ゼロ化
    W -= W.sum(axis=1, keepdims=True) / N
    return W