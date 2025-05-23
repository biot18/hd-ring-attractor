import numpy as np
import matplotlib.pyplot as plt
from config import N, T_total, dt
from network import build_vonmises_weights
from neuron import RateNetwork
from stimulus import bump_input, gaussian_noise


def run_simulation(noise_sigma=0.05, bump_center_speed=0.005):
    # 角度軸定義
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # 結合行列生成
    W = build_vonmises_weights()
    # ネットワーク初期化
    net = RateNetwork(W)

    # シミュレーションステップ数
    steps = int(T_total / dt)
    # 記録用配列
    records = np.zeros((steps, N))
    bump_center = 0.0

    for t in range(steps):
        # バンプ中心をゆっくり移動
        bump_center = (bump_center + bump_center_speed) % (2 * np.pi)
        # 外部入力生成
        I_bump = bump_input(theta, bump_center)
        I_noise = gaussian_noise(sigma=noise_sigma)
        I_ext = I_bump + I_noise
        
        # ネットワーク更新
        r, du = net.step(I_ext, return_du=True)
        
        
        # 異常検出
        if np.any(np.isnan(du)) or np.any(np.isinf(du)):
            print(f"du 異常 at t={t}: min={du.min()}, max={du.max()}")
            break
        # r 異常検出
        if np.any(~np.isfinite(r)):
            print(f"r 異常 at t={t}: min={np.nanmin(r)}, max={np.nanmax(r)}")
            break
        records[t] = r
        # チェック（最初の数ステップだけ出力）
        if t < 5:
            print(f"t={t}", 
              "I_ext:", np.min(I_ext), np.max(I_ext),
              "du:", np.min(du), np.max(du))
    records = np.nan_to_num(records, nan=0.0, posinf=0.0, neginf=0.0)
    return theta, records


if __name__ == '__main__':
    # シミュレーション実行
    theta, records = run_simulation()
    # 結合行列可視化例
    plt.figure(figsize=(6, 4))
    plt.imshow(records.T, aspect='auto', origin='lower',
               extent=[0, T_total, 0, 360])
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron angle (deg)')
    plt.title('Activity over time')
    plt.colorbar(label='Firing rate')
    plt.show()