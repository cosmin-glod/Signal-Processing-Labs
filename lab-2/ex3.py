import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wavfile
import time

# a
start = 0
end = 1
f = 400
nr_samples = 44_100

axis = np.linspace(start, end, nr_samples)
sinus = np.sin(2 * np.pi * f * axis)

sd.play(sinus, nr_samples)
sd.wait()
time.sleep(0.5)

semnal_d_16bit = np.int16(sinus * ((2**15) - 1))
wavfile.write("./sounds/sinus.wav", nr_samples, semnal_d_16bit)

# b
start = 0
end = 3
f = 800
nr_samples = 44_100

axis = np.linspace(start, end, nr_samples)
sinus = np.sin(2 * np.pi * f * axis)

sd.play(sinus, nr_samples)
sd.wait()
time.sleep(0.5)

# c
start = 0
end = 0.001
f = 240
nr_samples

axis = np.linspace(start, end, nr_samples)
sinus = np.sin(2 * np.pi * f * axis)
sawtooth = np.mod(np.arcsin(np.abs(sinus)), 0.1)

sd.play(sawtooth, nr_samples)
sd.wait()
time.sleep(0.5)


# d
start = 0
end = 0.01
nr_samples = 44_100
f = 300

axis = np.linspace(start, end, nr_samples)
sinus = np.sin(2 * np.pi * f * axis)
square = np.sign(sinus)

sd.play(square, 44_100)
sd.wait()
time.sleep(0.5)

# loading wav file from disk

fs, audio = wavfile.read("./sounds/sinus.wav")
sd.play(audio, fs)
sd.wait()