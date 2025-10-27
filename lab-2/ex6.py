import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 100
nr_samples = 42_100

a = 1
axis = np.linspace(start, end, nr_samples)
fs = 1 / (end - start)

f = [fs / 2, fs / 4, 0]

sinus = [np.sin(2 * np.pi * fq * axis) for fq in f]

plt.figure()
plt.suptitle("Three different frequencies")

plt.subplot(3, 1, 1)
plt.plot(axis, sinus[0])
plt.title("fs/2")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(axis, sinus[1])
plt.title("fs/4")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(axis, sinus[2])
plt.title("0 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig(fname="./plots/different_frequencies.pdf", format="pdf")

plt.show()