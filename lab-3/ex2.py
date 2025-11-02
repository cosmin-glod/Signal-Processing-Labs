import numpy as np
import matplotlib.pyplot as plt
import math

# animation settings
animation_step = 20
pause_time = 0.0001

start = 0
end = 1
nr_samples = 1000

axis = np.linspace(start, end, nr_samples)

f = 4
sinus = np.sin(2 * np.pi * f * axis)

# Figure 1
y = sinus * np.exp(-2 * np.pi * 1j * axis)

plt.ion()
fig1, axes1 = plt.subplots(figsize=(7, 7))

axes1.set_xlim(-1.1, 1.1)
axes1.set_ylim(-1.1, 1.1)
axes1.set_aspect('equal')
axes1.set_title("Figure 1")
axes1.set_xlabel("Real")
axes1.set_ylabel("Imaginary")
axes1.grid(True)

line1, = axes1.plot([], [], 'b-')
point1, = axes1.plot([], [], 'ro')

for i in range(animation_step, nr_samples + 1, animation_step):
    line1.set_data(y.real[:i], y.imag[:i])
    point1.set_data([y.real[i - 1]], [y.imag[i - 1]])
    plt.pause(pause_time)

plt.ioff()
plt.savefig(fname="./plots/fig1.pdf", format="pdf")
plt.show()

# Figure 2
omegas = [2, 7, 9, f]
zs = []
lines = []
points = []

plt.ion()
fig2, axes2 = plt.subplots(2, 2, figsize=(8, 8))

for o, ax in zip (omegas, axes2.flat):
        z = sinus * np.exp(-2 * np.pi * 1j * o * axis)
        zs.append(z)

        ax.set_xlim(-1.1, 1.1) 
        ax.set_ylim(-1.1, 1.1)

        ax.set_title(f"Omega = {o}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_aspect("equal")
        ax.grid(True)

        line2, = ax.plot([], [], 'b-')
        point2, = ax.plot([], [], 'ro')
        lines.append(line2)
        points.append(point2)

for i in range(animation_step, nr_samples + 1, animation_step):
    for o, ax, z, line2, point2 in zip(omegas, axes2.flat, zs, lines, points):
        line2.set_data(z.real[:i], z.imag[:i])
        point2.set_data([z.real[i - 1]], [z.imag[i - 1]])
        plt.pause(pause_time)

plt.tight_layout()
plt.savefig(fname="./plots/fig2.pdf", format="pdf")
plt.show()