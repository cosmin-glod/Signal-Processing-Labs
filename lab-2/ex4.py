import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
 
a = 2
f = 4
axis = np.linspace(0, 1, 42_100)

sinus = a * np.sin(2 * np.pi * f * axis)
sawtooth = saw = a * signal.sawtooth(2 * np.pi * f * axis)

plt.plot(axis, sinus + sawtooth)
plt.title("Sinus + sawtooth")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig(fname="./plots/sinus_plus_sawtooth.pdf", format="pdf")
plt.show()