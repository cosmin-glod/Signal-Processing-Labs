import numpy as np
import math
import matplotlib.pyplot as plt
 
N = 8

axis = np.linspace(0, N - 1, N)

fourier_matrix = np.zeros((N, N), dtype=np.complex64)

for a in range(N):
    for b in range(N):
        fourier_matrix[a][b] = np.exp(-2 * np.pi * 1j * b * (a / N))

fig, axes = plt.subplots(N, 2, figsize=(8, 8))
fig.suptitle('Fourier Matrix')

# for real numbers
for i in range(N):
    axes[i, 0].stem(axis, fourier_matrix[i].real)
    axes[i, 0].set_title(f"Real Line {i + 1}")
    axes[i, 0].grid(True)

# for imaginary numbers
for i in range(N):
    axes[i, 1].stem(axis, fourier_matrix[i].imag)
    axes[i, 1].set_title(f"Imaginary Line {i + 1}")
    axes[i, 1].grid(True)

plt.tight_layout()
plt.savefig(fname="./plots/fourier_line_by_line.pdf", format="pdf")
plt.show()

# -----

fourier_h = np.transpose(np.conjugate(fourier_matrix))
fourier_multiplication = np.dot(fourier_matrix, fourier_h)

identity_n = N * np.eye(N)
print(np.allclose(identity_n, fourier_multiplication, atol=1e-3))
