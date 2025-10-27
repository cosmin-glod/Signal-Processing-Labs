import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 1
f = 3
a = 10
nr_samples = 1000

axis = np.linspace(start, end, nr_samples)
sinus = a * np.sin(2 * np.pi * f * axis)
cosinus = a * np.cos(2 * np.pi * f * axis - np.pi / 2)

fig, axes = plt.subplots(2)
fig.suptitle("Sinus & Cosinus")

axes[0].plot(axis, sinus)
axes[0].set_title("Sinus Signal")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

axes[1].plot(axis, cosinus)
axes[1].set_title("Cosinus Signal")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (s)")

plt.tight_layout()

plt.savefig(fname="./plots/sin_and_cos.pdf", format="pdf")
plt.show()