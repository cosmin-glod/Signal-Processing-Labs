import numpy as np
import matplotlib.pyplot as plt

N = 20
def my_signal(t):
    # sum of 2 sin waves of different frequencies : 2 & 5
    return np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 5 * t) 

t_samples = np.linspace(0, 1, N, endpoint=False)
x = my_signal(t_samples)

d = 10
y = np.roll(x, d)

X = np.fft.fft(x)
Y = np.fft.fft(y)

a1 = np.fft.ifft(X.conj() * Y)
index1 = np.argmax(np.abs(a1))

eps = 1e-10
a2 = np.fft.ifft(Y / (X + eps)) # do not divide by 0
index2 = np.argmax(np.abs(a2))

print(f"Orignial Shift d = {d}")
print(f"Index 1 = {index1}")
print(f"Index 2 = {index2}")
print(f"The result difference between the 2 methods is {np.abs(index1 - index2)}")