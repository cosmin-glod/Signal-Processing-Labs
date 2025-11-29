import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta
import kagglehub

# Download latest version
dataset_path = kagglehub.dataset_download("lampubhutia/bullettrain-timeseries-data")
print("Path to dataset files:", dataset_path)

dataset = np.genfromtxt(dataset_path + '/Train.csv', delimiter=',')
dataset = dataset[1:, 2]

def d():
    fs = 1/3600
    dataset_fft = np.fft.fft(dataset)
    n = len(dataset) # n = 18288
    dataset_fft_normalized = np.abs(dataset_fft) / n

    f = fs * np.linspace(0, n // 2, n // 2) / n

    plt.figure(figsize=(12, 6))
    plt.plot(f, dataset_fft_normalized[:n//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of FFT')
    plt.grid(True)
    plt.savefig('./plots/abs_fft.pdf', format="pdf")
    plt.show()

def e(): # continuous component removal
    mean_value = np.mean(dataset)
    print(f"Mean value: {mean_value}")

    dataset_centered = dataset - mean_value
    print(f"The new mean is : {np.mean(dataset_centered)}") # it should be close to 0

    n = len(dataset)

    fft_original_mag = np.abs(np.fft.fft(dataset)) / n
    fft_centered_mag = np.abs(np.fft.fft(dataset_centered)) / n
    
    fs = 1/3600
    f = np.linspace(0, fs / 2, n // 2)

    plt.figure(figsize=(12, 6))
    
    zoom_samples = 20
    
    plt.plot(f[:zoom_samples], fft_original_mag[:zoom_samples], label='Original Signal', marker='o')
    plt.plot(f[:zoom_samples], fft_centered_mag[:zoom_samples], label='Centered Signal', marker='x')
    
    plt.title('Removing DC Component')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude |X(f)|')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname="./plots/removing_dc_component.pdf", format="pdf")
    plt.show()

def f():
    dataset_centered = dataset - np.mean(dataset)
    n = len(dataset)
    
    fft_mag = np.abs(np.fft.fft(dataset_centered)) / n
    fft_mag_half = fft_mag[:n // 2]
    
    top_indexes = np.argsort(fft_mag_half)[::-1]

    freqs_per_day = np.linspace(0, 24/2, n//2) # fs = 24 samples/day

    print("\nSubpunct f) Top 4 frequencies:")
    for i in range(10): # i wanted to see the 12h cycle and the 1W cycle
        idx = top_indexes[i]
        freq = freqs_per_day[idx]
        perioada_ore = 24 / freq if freq != 0 else 0
        
        print(f"{i+1}. Frequency: {freq:.3f} cycles/day | Period: {perioada_ore:.1f} hours | Magnitude: {fft_mag_half[idx]:.1f}")

def g():
    # the dataset starts on 25-aug-2012 (saturday)
    # first monday is on 27-aug-2012

    start_index = 48 # 2 days offset
    month_period = 24 * 30 # 30 days

    offset_weeks = 24 * 7 * 6 # 6 weeks offset

    current_start = start_index + offset_weeks

    dataset_month = dataset[current_start : current_start + month_period]

    dataset_start_date = datetime(2012, 8, 25, 0, 0)
    segment_start_date = dataset_start_date + timedelta(hours=int(current_start))
    dates = [segment_start_date + timedelta(hours=i) for i in range(len(dataset_month))]

    plt.figure(figsize=(15, 6))
    plt.plot(dates, dataset_month)

    plt.xlabel('Time (Days)')
    plt.ylabel('Number of cars')
    plt.title('One month of traffic (Starting on a Monday)')
    plt.grid(True)
    plt.savefig('./plots/one_month_of_traffic.pdf')
    plt.show()


def i():
    n = len(dataset)
    
    fft_complex = np.fft.fft(dataset)
    
    days_total = n / 24 
    cutoff_freq_cycles_per_day = 4 
    cutoff_index = int(cutoff_freq_cycles_per_day * days_total)
    
    print("-" * 50)
    print(f"Filtering details:")
    print(f"  Cutoff Index: {cutoff_index} (frequencies kept out of {n//2})")
    
    fft_filtered = fft_complex.copy()
    fft_filtered[cutoff_index : n - cutoff_index] = 0
    
    dataset_filtered = np.fft.ifft(fft_filtered).real

    t = np.linspace(0, days_total, n)

    start_zoom = 24 * 7 * 6 + 48 
    end_zoom = start_zoom + (24 * 14) 
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t[start_zoom:end_zoom], dataset[start_zoom:end_zoom], color='red', linewidth=1.5)
    plt.title('Original Signal (2 Weeks)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Cars')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t[start_zoom:end_zoom], dataset_filtered[start_zoom:end_zoom], color='blue', linewidth=2)
    plt.title(f'Filtered Signal (2 Weeks)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Cars')
    plt.legend() 
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./plots/filtered_signal.pdf')
    plt.show()