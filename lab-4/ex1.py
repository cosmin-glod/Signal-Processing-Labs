import numpy as np
import matplotlib.pyplot as plt
import time

Ns = [2**x for x in range(7, 14)]

times_my_dft = []
times_np_dft = []
times_my_fft = []

def my_fft(x):
    N = len(x)
    if N == 1:
        return x
    
    # divide et impera
    x_even = x[::2]
    x_odd = x[1::2]
    
    X_even = my_fft(x_even)
    X_odd = my_fft(x_odd)

    half_N = N // 2
    factor = np.exp(-2j * np.pi * np.arange(half_N) / N)
    
    X_first_half = X_even + factor * X_odd
    X_second_half = X_even - factor * X_odd

    X = np.concatenate([X_first_half, X_second_half])
    
    return X

for N in Ns:
    x = np.random.rand(N) # random signal

    # my DFT
    start_time = time.perf_counter()

    indices = np.arange(N)
    ab_matrix = np.outer(indices, indices)
    const = -2 * np.pi * 1j / N
    fourier_matrix = np.exp(const * ab_matrix)
    X_my_dft = np.dot(fourier_matrix, x)

    end_time = time.perf_counter()
    times_my_dft.append(end_time - start_time)

    # my FFT
    start_time = time.perf_counter()
    X_my_fft = my_fft(x)
    end_time = time.perf_counter()
    times_my_fft.append(end_time - start_time)

    # numpy FFT
    start_time = time.perf_counter()
    X_np_fft = np.fft.fft(x)
    end_time = time.perf_counter()

    times_np_dft.append(end_time - start_time)

times_my_dft = np.multiply(times_my_dft, 1000)
times_my_fft = np.multiply(times_my_fft, 1000)
times_np_dft = np.multiply(times_np_dft, 1000)

plt.figure(figsize=(10, 6))
plt.title("My DFT/FFT vs Numpy FFT times")
plt.plot(Ns, times_my_dft, 'o-', label="My DFT")
plt.plot(Ns, times_my_fft, 'o-', label="My FFT")
plt.plot(Ns, times_np_dft, 'o-', label="NumPy FFT")
plt.ylabel("Time (ms)")
plt.xlabel("Matrix Dimension")
plt.yscale("log")
plt.xticks(Ns, labels=[str(n) for n in Ns], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(fname="./plots/my_dft_vs_my_fft_vs_numpy_fft.pdf", format="pdf")
plt.show()