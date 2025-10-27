import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 1
nr_samples = 1000

axis = np.linspace(start, end, nr_samples)

a = 1
f = 5
faza = [1, 4, 9, 0.1]
sinus = [a * np.sin(2 * np.pi * f * axis + fz) for fz in faza]

fig, axes = plt.subplots(len(faza))
fig.suptitle("Sinus Signal - 5 Phases")

for i in range(len(faza)):
    axes[i].plot(axis, sinus[i])
    # axes[i].set_title(f"Sinus Signal {i + 1}")
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Amplitude")

plt.tight_layout()
plt.grid(True)

plt.savefig(fname="./plots/5_different_phases.pdf", format="pdf")
plt.show()

# ----------

SNR = [0.1, 1, 10, 100]

x = sinus[0]
z = np.random.normal(size=(nr_samples,))

x_norm = np.linalg.norm(x)
z_norm = np.linalg.norm(z)
gama = [np.sqrt((x_norm/z_norm)**2 / snr) for snr in SNR]

new_signals = [x + gama[i] * z for i in range(4)]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Noisy Signals with SNR = [0.1, 1, 10, 100]')

for i in range(len(new_signals)):
    axes[i // 2, i % 2].plot(axis, new_signals[i])
    axes[i // 2, i % 2].set_title(f"SNR = {SNR[i]}")
    axes[i // 2, i % 2].set_xlabel("Time (s)")
    axes[i // 2, i % 2].set_ylabel("Amplitude")
    axes[i // 2, i % 2].grid(True)

plt.grid(True)
plt.tight_layout()
plt.savefig(fname="./plots/noisy_signals_with_different_SNR.pdf", format="pdf")
plt.show()
