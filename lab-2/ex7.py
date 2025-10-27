import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 0.1
sample_rate = 1000
nr_samples = int(sample_rate * (end - start))

axis = np.linspace(start, end, nr_samples)

a = 1
f = 2

# a

sinus = np.sin(2 * np.pi * f * axis)

axis_reduced1 = axis[3::4]
sinus_reduced1 = sinus[3::4]

plt.figure(figsize=(6, 6))
plt.suptitle(f"{nr_samples} vs {nr_samples // 4} samples (every 4th sample) - same signal")

plt.subplot(2, 1, 1)
plt.stem(axis, sinus)
plt.title(f"{nr_samples} samples")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(axis_reduced1, sinus_reduced1)
plt.title(f"{nr_samples // 4} samples (every 4th)")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.savefig(fname="./plots/every_4th_sample.pdf", format="pdf")
plt.show()

# b

sinus_reduced2 = sinus[2::4]
axis_reduced2 = axis[2::4]

plt.figure(figsize=(6, 6))
plt.suptitle(f"{nr_samples} vs {nr_samples // 4} samples (every 4th / 2nd sample) - same signal")

plt.subplot(3, 1, 1)
plt.stem(axis, sinus)
plt.title(f"{nr_samples} samples")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(axis_reduced1, sinus_reduced1)
plt.title(f"{nr_samples // 4} samples (every 4th)")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(axis_reduced2, sinus_reduced2)
plt.title(f"{nr_samples // 4} samples (every 2nd)")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)


plt.tight_layout()
plt.savefig(fname="./plots/every_4th_and_2nd_sample.pdf", format="pdf")
plt.show()