import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

FS = 12000  # sampling rate (Hz)


def main():
    parser = argparse.ArgumentParser(description="Plot CWRU healthy vs faulty signal")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="folder containing the .mat files (default: data)",
    )
    args = parser.parse_args()

    healthy_data = scipy.io.loadmat(os.path.join(args.data_dir, '97.mat'))
    faulty_data = scipy.io.loadmat(os.path.join(args.data_dir, '105.mat'))

    healthy_signal = healthy_data['X097_DE_time'][:1000].flatten()
    faulty_signal = faulty_data['X105_DE_time'][:1000].flatten()
    t_ms = np.arange(1000) / FS * 1000  # time axis in milliseconds

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_ms, healthy_signal, color='green')
    plt.title('Healthy Bearing - Drive-End Vibration (Time Domain)')
    plt.ylabel('Acceleration (g)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_ms, faulty_signal, color='red')
    plt.title('Inner Race Fault - Drive-End Vibration (Time Domain)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (g)')
    plt.grid(True)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(os.path.join('outputs', 'fig1_time_domain.png'), dpi=200)
    print("Saved outputs/fig1_time_domain.png")
    plt.show()


if __name__ == "__main__":
    main()