import numpy as np
import matplotlib.pyplot as plt

def generator_sinus(a, f, t, faza):
    return a * np.sin(2 * np.pi * f * t + faza)

start = 0
end = 1
f = 9
fs = 10 # (< 2 * f)

t = np.linspace(start, end, 2000)
ts = np.linspace(start, end, fs, endpoint=False)

sin1 = generator_sinus(1, f, t, 0)
sin1_sampled = generator_sinus(1, f, ts, 0)

fig, axs = plt.subplots(4, 1, figsize=(10, 6))
fig.suptitle("Fenomenul de aliere")

axs[0].plot(t, sin1)
axs[0].set_title(f"Original Signal with {f}Hz")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency")
axs[0].grid(True)

axs[1].plot(t, sin1)
axs[1].scatter(ts, sin1_sampled, color="black")
axs[1].set_title(f"Sampled Original Signal with {f}Hz - sample_rate={fs}Hz")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

f_new = f + fs
sin2 = generator_sinus(1, f_new, t, 0)
sin2_sampled = generator_sinus(1, f_new, ts, 0)

axs[2].plot(t, sin2)
axs[2].scatter(ts, sin2_sampled, color="black")
axs[2].set_title(f"Signal with {f_new}Hz - sample_rate={fs}Hz")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Frequency")
axs[2].grid(True)

f_new = f - fs
sin3 = generator_sinus(1, f_new, t, 0)
sin3_sampled = generator_sinus(1, f_new, ts, 0)

axs[3].plot(t, sin3)
axs[3].scatter(ts, sin3_sampled, color="black")
axs[3].set_title(f"Signal with {f_new}Hz - sample_rate={fs}Hz")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Frequency")
axs[3].grid(True)

plt.tight_layout()
plt.savefig(fname="./plots/fenomenul_de_aliere.pdf", format="pdf")
plt.show()