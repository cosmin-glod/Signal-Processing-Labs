import numpy as np
import matplotlib.pyplot as plt

N = 10

p = np.random.randint(-20, 20, size = N + 1)
q = np.random.randint(-10, 10, size = N + 1)

ans_len = len(p) + len(q) - 1

# Direct Polynomial Multiplication
ans_direct = np.zeros(ans_len, dtype=float)
for i in range(len(p)):
    for j in range(len(q)):
        ans_direct[i + j] += p[i] * q[j]

# FFT (Convolution Theorem)
fft_size = 1
while fft_size < ans_len:
    fft_size *= 2

P = np.fft.fft(p, n=fft_size)
Q = np.fft.fft(q, n=fft_size)

result_freq_domain = P * Q
ans_fft = np.fft.ifft(result_freq_domain)
ans_fft = np.round(ans_fft.real[:ans_len])

print('p(x): ', p)
print('q(x): ', q)
print()
print('r(x) direct method: \n', ans_direct)
print()
print('r(x) FFT method: \n', ans_fft)

is_correct = np.allclose(ans_direct, ans_fft)
if is_correct:
    print("✅ The results match")
else:
    print("❌ The results are different")