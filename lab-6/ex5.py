import numpy as np
import matplotlib.pyplot as plt

def rectangularWindow(total_len, win_len, start_index):
    window_array = np.zeros(total_len)
    core_window = np.ones(win_len)

    end_index = min(start_index + win_len, total_len)
    actual_len = end_index - start_index

    window_array[start_index:end_index] = core_window[:actual_len]
    return window_array

def hanningWindow(total_len, win_len, start_index):
    window_array = np.zeros(total_len)

    n = np.arange(win_len)
    core_window = 0.5 * (1 - np.cos(2 * np.pi * n / (win_len - 1)))

    end_index = min(start_index + win_len, total_len)
    actual_len = end_index - start_index

    window_array[start_index:end_index] = core_window[:actual_len]
    return window_array

f = 100
A = 1
phi = 0

Nw = 200
N_signal = 500
Fs = 1000.0
window_start_pos = 100

n = np.arange(N_signal)
time = n / Fs
sinus = A * np.sin(2 * np.pi * f * time)

rectWindow = rectangularWindow(N_signal, Nw, window_start_pos)
hannWindow = hanningWindow(N_signal, Nw, window_start_pos)

rectSignal = sinus * rectWindow
hannSignal = sinus * hannWindow

plt.subplot(3, 1, 1)
plt.plot(time, sinus)
plt.title('Original sine wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, rectSignal)
plt.title('Rectangular window')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, hannSignal)
plt.title('Hanning window')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.savefig('./plots/rect_vs_hanning.pdf')
plt.show()