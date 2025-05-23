import numpy as np
from config import N

def circular_distance_matrix(N=N):
    """ニューロン間の環状距離行列を返す"""
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    dtheta = np.abs(theta[:, None] - theta[None, :])
    return np.minimum(dtheta, 2*np.pi - dtheta)


def normalize_rows(W):
    """行和をゼロに正規化"""
    W = W.copy()
    W -= W.sum(axis=1, keepdims=True) / W.shape[1]
    return W


def save_numpy(path, array):
    """NumPy 配列を保存"""
    np.save(path, array)


def load_numpy(path):
    """NumPy 配列を読み込み"""
    return np.load(path)