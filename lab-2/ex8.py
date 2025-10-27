import numpy as np
import matplotlib.pyplot as plt

start = -np.pi / 2
end = np.pi / 2
nr_samples = 10_000

axis = np.linspace(start, end, nr_samples)

sinus = np.sin(axis)
aprox_liniar = axis
aprox_pade = (axis - (7 * axis ** 3) / 60) / (1 + (axis ** 2) / 20)

plt.figure(figsize=(7, 6))
plt.title("Sinus vs Liniar Aproximation")
plt.plot(axis, sinus, label="Sinus function")
plt.plot(axis, aprox_liniar, label="Liniar Aproximation", linestyle="--")
plt.plot(axis, aprox_pade, label="Pade Aproximation", linestyle="--")

plt.xlabel("Radians")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fname="./plots/sinus_and_different_aproximations.pdf", format="pdf")
plt.show()

# Error Graphs

error_liniar = np.abs(sinus - aprox_liniar)
error_pade = np.abs(sinus - aprox_pade)

plt.figure(figsize=(7, 6))
plt.title("Errors for liniar and pade aproximations (smaller is better)")
plt.yscale("log")
plt.plot(axis, error_liniar, label="|sin(x) - x|")
plt.plot(axis, error_pade, label="|sin(x) - pade(x)|")
plt.xlabel("Radians")
plt.ylabel("Error (log scale)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(fname="./plots/errors_for_liniar_and_pade_compared_to_sinus.pdf", format="pdf")
plt.show()