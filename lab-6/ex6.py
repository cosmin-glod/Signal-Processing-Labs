import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub
from scipy import signal

# Download latest version
path = kagglehub.dataset_download("lampubhutia/bullettrain-timeseries-data")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/Train.csv")
full_signal = df.iloc[:, 2].values

# a
start_index = 6700 # arbitrary index
x = full_signal[start_index : start_index+24*3]

# b
w_list = [5, 9, 13, 17]

plt.figure(figsize=(12, 7))
plt.plot(x, label="Original data", c = "black")
for w in w_list:
    filter = np.ones(w) / w
    filtered_signal = np.convolve(x, filter, mode='same') # same handles edges
    plt.plot(filtered_signal, label=f"Window size = {w}", linestyle='--')

plt.title("Moving Average Filtering")
plt.xlabel("Time (hours)")
plt.ylabel("No. Cars")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(fname="./plots/moving_average.pdf", format="pdf")
plt.show()

# c
fs = 1.0 # 1 sample per hour
nyquist = fs / 2

# Cuts the events faster than 5 hours
cutoff_period_hours = 5
fc = 1.0 / cutoff_period_hours

Wn = fc / nyquist

# d
N_order = 5 # Filter order
rp = 5

b_butter, a_butter = signal.butter(N_order, Wn, btype='low')
b_cheby, a_cheby = signal.cheby1(N_order, rp, Wn, btype='low')

# e
y_butter = signal.filtfilt(b_butter, a_butter, x)
y_cheby = signal.filtfilt(b_cheby, a_cheby, x)

plt.figure(figsize=(12, 8))

plt.plot(x, label="Original Signal")
plt.plot(y_butter, label="Butterworth")
plt.plot(y_cheby, label=f"Chebyshev - rp = {rp}")

plt.title(f"Filter Comparison (Cutoff = {cutoff_period_hours})")
plt.xlabel("Time (hours)")
plt.ylabel("No. Cars")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/filter_comparison.pdf")
plt.show()

# f
orders = [1, 4, 15]

plt.figure(figsize=(12, 6))
plt.plot(x, label="Original signal", linewidth=3, c="black")

for N in orders:
    b, a = signal.butter(N, Wn, btype="low")
    y_filt = signal.filtfilt(b, a, x)
    plt.plot(y_filt, label=f"Butterworth Ordin {N}")

plt.title(f"The effect of the filter order")
plt.xlabel("Time (hours)")
plt.ylabel("No. Cars")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/filter_order_comparison.pdf")
plt.show()


rp_list = [0.1, 10, 20]

plt.figure(figsize=(12, 6))
plt.plot(x, label="Original")

for rp in rp_list:
    b, a = signal.cheby1(N_order, rp, Wn, btype='low')
    y_filt = signal.filtfilt(b, a, x)

    plt.plot(y_filt, label=f'Chebyshev Ripple = {rp} dB')

plt.title("Ripple Effect for Chebyshev (Order = 5)")
plt.xlabel('Time (hours)')
plt.ylabel('No. Cars')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/ripple_filter_comparison.pdf")
plt.show()