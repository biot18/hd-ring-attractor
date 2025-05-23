import matplotlib.pyplot as plt
import numpy as np

def plot_tuning_curve(theta, rate, params=None):
    plt.figure()
    plt.plot(np.degrees(theta), rate, label='data')
    if params is not None:
        mu, kappa, A, B = params
        plt.plot(np.degrees(theta), A*np.exp(kappa*np.cos(theta-mu))+B, '--', label='fit')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Firing rate')
    plt.legend()
    plt.show()


def animate_activity(records, dt, interval=50):
    import matplotlib.animation as animation
    theta = np.linspace(0, 360, records.shape[1])
    fig, ax = plt.subplots()
    line, = ax.plot(theta, records[0])
    ax.set_ylim(0, records.max())

    def update(i):
        line.set_ydata(records[i])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(records), interval=interval)
    plt.show()