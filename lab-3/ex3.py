import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 1
nr_samples = 1000

axis = np.linspace(start, end, nr_samples)

f1, a1 = 5, 3
f2, a2 = 30, 6
f3, a3 = 15, 9

sin1 = a1 * np.sin(2 * np.pi * f1 * axis)
sin2 = a2 * np.sin(2 * np.pi * f2 * axis)
sin3 = a3 * np.sin(2 * np.pi * f3 * axis)

signal = sin1 + sin2 + sin3

max_omega = 50
omega_range = np.arange(0, max_omega + 1)

N = nr_samples

fourier_matrix = np.zeros((N, N), dtype=np.complex64)

for a in range(nr_samples):
    for b in range(nr_samples):
        fourier_matrix[a][b] = np.exp(-2 * np.pi * 1j * b * (a / nr_samples))
    
X_transform_full = fourier_matrix @ signal
X_magnitude_full = np.abs(X_transform_full)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(axis, signal)
ax1.set_title(f"Summed signals: f = {f1}, {f2}, {f3} Hz")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude x(t)")
ax1.grid(True)
ax1.set_xlim(0, 0.5) 


ax2.stem(omega_range, X_magnitude_full[:max_omega + 1])
ax2.set_title("|X(omaga)| (First 100 omegas)")
ax2.set_xlabel("Frequency(omaga in Hz)")
ax2.set_ylabel("|X(omega)|")
ax2.grid(True)

plt.tight_layout()
plt.savefig(fname="./plots/ex3.pdf", format="pdf")
plt.show()