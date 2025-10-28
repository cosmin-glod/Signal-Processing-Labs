import numpy as np
import matplotlib.pyplot as plt
import math

start = 0
end = 1
nr_samples = 1000

axis = np.linspace(start, end, nr_samples)

# Figure 1

f = 5
sinus = np.sin(2 * np.pi * f * axis)
y = sinus * np.exp(-2 * np.pi * 1j * axis)

plt.figure(figsize=(10, 6))
plt.plot(y.real, y.imag)
plt.title("Figure 1")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.savefig(fname="./plots/fig1.pdf", format="pdf")
plt.show()

# Figure 2

