import numpy as np
import matplotlib.pyplot as plt

# a
start = 0
end = 1
f = 400
nr_samples = 1600

axis = np.linspace(start, end, nr_samples)
sinus = np.sin(2 * np.pi * f * axis)

plt.stem(axis, sinus)
plt.title("Sinusoidal Signal - 1600 Samples")
plt.savefig("./plots/sin_400hz_1600_samples.pdf", format="pdf")
plt.show()

# b
start = 0
end = 3
f = 800
nr_samples = 100

axis = np.linspace(start, end, nr_samples)
sinus = np.sin(2 * np.pi * f * axis)
plt.stem(axis, sinus)
plt.title("Sinusoidal Signal 800Hz - 100 Samples")
plt.savefig("./plots/sin_800hz_100_samples.pdf", format="pdf")
plt.show()

# c
start = 0
end = 0.001
f = 240

axis = np.linspace(start, end, 1000)
sinus = np.sin(2 * np.pi * f * axis)
sawtooth = np.mod(np.arcsin(np.abs(sinus)), 0.1)

plt.plot(axis, sawtooth)
plt.title("Sawtooth signal - 240Hz")
plt.savefig("./plots/sawtooth_240hz.pdf", format="pdf")
plt.show()

# d
start = 0
end = 0.01
nr_samples = 1000
f = 300

axis = np.linspace(start, end, nr_samples)
sinus = np.sin(2 * np.pi * f * axis)
square = np.sign(sinus)

plt.plot(axis, square)
plt.title("Square signal - 300Hz")
plt.savefig("./plots/square_300hz.pdf", format="pdf")
plt.show()

# e
random_signal = np.random.rand(128, 128)
plt.imshow(random_signal)
plt.title("Random Image")
plt.savefig(fname="./plots/random_image.pdf", format="pdf")
plt.show()

# f
generated = np.zeros((128, 128))

generated[:, ::2] = 128
generated[1::2, :] = 128

plt.imshow(generated)
plt.title("Chess Like Image")
plt.savefig(fname="./plots/chess_like_image.pdf", format="pdf")
plt.show()