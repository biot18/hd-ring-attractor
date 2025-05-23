import numpy as np
from config import dt, tau, max_r

#leak項を追加し発散を防ぐ
leak_alpha = 1.0

class RateNetwork:
    def __init__(self, W):
        self.W = W
        self.N = W.shape[0]
        self.u = np.zeros(self.N)
        self.r = np.zeros(self.N)

    def step(self, I_ext, return_du=False):
        """
        1ステップ更新（Euler 法）
        return_du=True のとき、(r, du) を返す
        """
        du = (-(1.0+leak_alpha)*self.u + self.W.dot(self.r) + I_ext) * (dt / tau)
        self.u += du
        self.r = np.clip(self.u, 0, max_r)
        if return_du:
            return self.r, du
        return self.r