import numpy as np
import matplotlib.pyplot as plt

# a
axis = np.linspace(0.0, 0.03, 600)

# b
def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)
def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)
def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

fig, axes = plt.subplots(3)
fig.suptitle("Signal Graphs")

axes[0].plot(axis, x(axis))
axes[1].plot(axis, y(axis))
axes[2].plot(axis, z(axis))
plt.savefig(fname="./plots/signal_graphs.pdf", format="pdf")
plt.show()

# c

freq = 200
start = 0
end = 0.03
nr_samples = int(freq * (end - start))

samples = np.linspace(start, end, nr_samples)

fig, axes = plt.subplots(3)
fig.suptitle("Sampled Signal Graphs")

axes[0].stem(samples, x(samples))
axes[1].stem(samples, y(samples))
axes[2].stem(samples, z(samples))
plt.savefig(fname="./plots/sampled_signal_samples.pdf", format="pdf")
plt.show()