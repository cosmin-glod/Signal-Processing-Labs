import numpy as np
import matplotlib.pyplot as plt

N = 100
no_iterations = 3

# random signal
x_signal = np.random.rand(N)

signals = [x_signal]
x_current = x_signal

for _ in range(no_iterations):
    x_current = np.convolve(x_current, x_current)
    signals.append(x_current)


plt.figure(figsize=(12, 7))
    
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(signals[i])
    plt.title(f"Iteration {i} - Len = {len(signals[i])}")

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
plt.tight_layout()
plt.savefig(fname=f"./plots/ex2_random.pdf", format="pdf")
plt.show()


# rectangle signal

rectangle_signal = np.zeros(N)
rectangle_signal[N//4:N//2] = 1.0


signals = [rectangle_signal]
x_current = rectangle_signal

for _ in range(no_iterations):
    x_current = np.convolve(x_current, x_current)
    signals.append(x_current)


plt.figure(figsize=(12, 7))
    
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(signals[i])
    plt.title(f"Iteration {i} - Len = {len(signals[i])}")

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
plt.tight_layout()
plt.savefig(fname=f"./plots/ex2_rectangle.pdf", format="pdf")
plt.show()

