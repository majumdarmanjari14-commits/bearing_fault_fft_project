import os
import numpy as np
import scipy.io
import scipy.fft
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

plt.style.use('default')

healthy_data = scipy.io.loadmat(os.path.join(DATA_DIR, '97.mat'))
faulty_data = scipy.io.loadmat(os.path.join(DATA_DIR, '105.mat'))

# Keep the full signal and flatten (n, 1) -> (n,)
healthy_signal = healthy_data['X097_DE_time'].flatten()
faulty_signal = faulty_data['X105_DE_time'].flatten()

fs = 12000
N = 12000  # 1 second window -> 1 Hz resolution

def calculate_fft(signal):
    segment = signal[:N]
    clean_signal = segment - np.mean(segment)
    fft_output = scipy.fft.fft(clean_signal)
    freqs = scipy.fft.fftfreq(N, 1/fs)[:N//2]
    magnitudes = np.abs(fft_output[:N//2]) * (2 / N)
    return freqs, magnitudes

freqs, healthy_mag = calculate_fft(healthy_signal)
_, faulty_mag = calculate_fft(faulty_signal)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, facecolor='white')

ax1.plot(freqs, healthy_mag, color='#00802b', linewidth=1.2, label='Healthy')
ax1.set_ylabel('Amplitude (g)')
ax1.grid(True, color='#e0e0e0', linestyle='--')
ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc')

ax2.plot(freqs, faulty_mag, color='#d60036', linewidth=1.2, label='Inner Race Fault')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude (g)')
ax2.grid(True, color='#e0e0e0', linestyle='--')
ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc')

ax2.set_xlim(5, 600)

bpfi = 5.4152 * 1797 / 60  # inner-race fault frequency, ~162.2 Hz
for ax in (ax1, ax2):
    ax.axvline(bpfi, color='black', linestyle='--', linewidth=0.8)
ax2.text(bpfi + 5, ax2.get_ylim()[1] * 0.8, f'BPFI = {bpfi:.1f} Hz', fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, 'outputs', 'fig2_fft.png'), dpi=200)
print("Saved outputs/fig2_fft.png")
plt.show()
