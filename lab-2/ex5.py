import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wavfile

start = 0
end = 1
f1 = 400
f2 = 800
nr_samples = 44_100

axis = np.linspace(start, end, nr_samples)

sinus1 = np.sin(2 * np.pi * f1 * axis)
sinus2 = np.sin(2 * np.pi * f2 * axis)

semnal = np.concatenate((sinus1, sinus2))

sd.play(semnal, nr_samples)
sd.wait()

semnal_d_16bit = np.int16(semnal * ((2**15) - 1))
wavfile.write("./sounds/two_frequencies.wav", nr_samples, semnal_d_16bit)

# Observations: there are 2 notes, the second one is higher than the first onw