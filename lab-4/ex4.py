import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 1

def generator_sinus(a, f, t, phase):
    return a * np.sin(2 * np.pi * f * t + phase)

def get_signal(t):
    return generator_sinus(1, 40, t, 0) + generator_sinus(1, 200, t, 0) + generator_sinus(1, 100, t, 0) + generator_sinus(1, 150, t, 0)

t_continuous = np.linspace(0, 1, 2000)
ts_correct = np.linspace(0, 1, 400, endpoint=False)
ts_incorrect = np.linspace(0, 1, 100, endpoint=False)

signal = get_signal(t_continuous)
signal_sampled_correct = get_signal(ts_correct)
signal_sampled_incorrect = get_signal(ts_incorrect)

signal_150hz = generator_sinus(1, 150, t_continuous, 0)

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Sampling Comparison')

axs[0].plot(t_continuous, signal, color='c', label='Bass Signal')
axs[0].scatter(ts_correct, signal_sampled_correct, color='darkgreen')
axs[0].plot(t_continuous, signal_150hz, color='red', label='150 Hz signal')
axs[0].set(xlabel='Time (s)', ylabel='Amplitude', title='Bass Signal Sampled Correctly (fs >= 2B)')
axs[0].legend()

axs[1].plot(t_continuous, signal, color='c', label='bass signal')
axs[1].scatter(ts_incorrect, signal_sampled_incorrect, color='darkgreen')
axs[1].plot(t_continuous, signal_150hz, color='red', label='150 Hz signal')
axs[1].set(xlabel='Time (s)', ylabel='Amplitude', title='Bass Signal Sampled Incorrectly (fs < 2B)')
axs[1].legend()

plt.savefig(fname="./plots/bass.pdf")
plt.tight_layout()
plt.show()