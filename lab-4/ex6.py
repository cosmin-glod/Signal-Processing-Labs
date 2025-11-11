import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wv

freq, vocale = wv.read('vowels.wav')

n = len(vocale)

window_size = int(0.01 * n)
step = int(window_size * 0.5)

windows = []

for i in range(0, n - window_size, step):
    val = vocale[i:i+window_size]
    
    val_fft = np.abs(np.fft.rfft(val))
    
    windows.append(val_fft)

spect = np.column_stack(windows)
frq = np.fft.rfftfreq(window_size, d=1.0 / freq)

times = np.arange(spect.shape[1]) * (step / float(freq))

spect_db = 20 * np.log10(spect + 1e-9)

plt.figure(figsize=(12, 8))
plt.pcolormesh(times, frq, spect_db, cmap='plasma', shading='gouraud')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Vowels Spectogram')
plt.colorbar(label='Power (dB)')

plt.ylim(0, 8000) 

plt.savefig(fname="./plots/spectograma.pdf")
plt.tight_layout()
plt.show()