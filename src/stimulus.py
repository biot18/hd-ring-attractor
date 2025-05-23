import numpy as np
from config import N, I0_bump, kappa_bump


def bump_input(theta, center, I0=I0_bump, kappa=kappa_bump):
    """
    環状上のバンプ刺激を生成
    Parameters
    ----------
    theta : array_like
        各ニューロンの角度 (rad)
    center : float
        バンプ中心角度 (rad)
    Returns
    -------
    I_ext : ndarray, shape (N,)
    """
    d = np.abs(theta - center)
    d = np.minimum(d, 2*np.pi - d)
    return I0 * np.exp(kappa * np.cos(d))


def gaussian_noise(N=N, sigma=0.1):
    """
    ガウスノイズを生成
    """
    return np.random.normal(0, sigma, size=N)