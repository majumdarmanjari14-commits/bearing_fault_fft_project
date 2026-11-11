"""
Step 4 - Automated bearing fault detection (CWRU drive-end data)

Pipeline for each recording:
  1. Load the drive-end signal and shaft speed (RPM) from the .mat file
  2. Compute the bearing characteristic fault frequencies (BPFI, BPFO, BSF)
  3. Time-domain features: RMS, peak, crest factor, kurtosis
  4. Hann-windowed FFT spectrum (raw)
  5. Envelope spectrum: band-pass -> Hilbert transform -> FFT
  6. Peak detection (scipy.signal.find_peaks) and matching against
     the fault frequencies -> automatic diagnosis

Outputs (saved in ./outputs):
  fig3_spectrum_comparison.png, fig4_envelope_spectra.png,
  fig5_time_features.png, results.csv
"""

import os
import csv
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FS = 12000                 # sampling rate of the 12k drive-end data (Hz)
BAND = (2000, 5000)        # resonance band used for envelope analysis (Hz)
TOL = 3.0                  # matching tolerance (Hz)
SNR_THRESHOLD = 30.0       # a fault line must stand >30x above the envelope-spectrum median

# SKF 6205-2RS JEM deep-groove ball bearing (drive end), multiples of shaft speed
FAULT_MULTIPLIERS = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 2.3568}
FAULT_NAMES = {"BPFI": "Inner Race Fault", "BPFO": "Outer Race Fault",
               "BSF": "Ball Fault"}

FILES = {
    "Healthy": "97.mat",
    "Inner Race Fault": "105.mat",
    "Ball Fault": "118.mat",
    "Outer Race Fault": "130.mat",
}


def load_signal(filename):
    """Return (drive-end signal as 1-D array, shaft speed in RPM)."""
    data = scipy.io.loadmat(os.path.join(DATA_DIR, filename))
    de_key = [k for k in data if k.endswith("DE_time")][0]
    rpm_key = [k for k in data if k.endswith("RPM")][0]
    signal = data[de_key].flatten()
    rpm = float(np.squeeze(data[rpm_key]))
    return signal, rpm


def fault_frequencies(rpm):
    shaft_hz = rpm / 60.0
    return {name: m * shaft_hz for name, m in FAULT_MULTIPLIERS.items()}


def time_features(x):
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    return {"rms": rms, "peak": peak, "crest": peak / rms,
            "kurtosis": kurtosis(x, fisher=False)}


def spectrum(x):
    """Single-sided amplitude spectrum with a Hann window."""
    n = len(x)
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft((x - x.mean()) * w)) * 2 / w.sum()
    freqs = np.fft.rfftfreq(n, 1 / FS)
    return freqs, mag


def envelope_spectrum(x):
    """Band-pass around the structural resonance, take the Hilbert
    envelope, then return the spectrum of that envelope."""
    b, a = butter(4, BAND, btype="band", fs=FS)
    env = np.abs(hilbert(filtfilt(b, a, x)))
    return spectrum(env - env.mean())


def diagnose(freqs, env_mag, ff):
    """Match envelope-spectrum peaks to the fault frequencies (and their
    2nd harmonic). Returns the diagnosis and the SNR for each fault type."""
    band = (freqs > 10) & (freqs < 500)
    noise_floor = np.median(env_mag[band])
    peaks, _ = find_peaks(env_mag, height=SNR_THRESHOLD * noise_floor)
    peak_freqs = freqs[peaks]

    snr = {}
    for name, f0 in ff.items():
        best = 0.0
        for h in (1, 2):
            window = np.abs(freqs - h * f0) <= TOL
            best = max(best, env_mag[window].max() / noise_floor)
        snr[name] = best

    matched = {n: s for n, s in snr.items()
               if s >= SNR_THRESHOLD and
               np.any(np.abs(peak_freqs - ff[n]) <= TOL) |
               np.any(np.abs(peak_freqs - 2 * ff[n]) <= TOL)}
    if not matched:
        return "Healthy", snr
    return FAULT_NAMES[max(matched, key=matched.get)], snr


def main():
    results = []
    plot_data = {}
    print(f"{'Recording':<18}{'RPM':>6}{'RMS':>8}{'Kurt':>7}"
          f"{'SNR_BPFI':>10}{'SNR_BPFO':>10}{'SNR_BSF':>9}  Diagnosis")
    print("-" * 92)
    for label, fname in FILES.items():
        x, rpm = load_signal(fname)
        n = (len(x) // FS) * FS            # use whole seconds of data
        x = x[:n]
        ff = fault_frequencies(rpm)
        feats = time_features(x)
        f_raw, m_raw = spectrum(x)
        f_env, m_env = envelope_spectrum(x)
        verdict, snr = diagnose(f_env, m_env, ff)
        plot_data[label] = (f_raw, m_raw, f_env, m_env, ff)
        row = {"recording": label, "file": fname, "rpm": rpm,
               **{k: round(v, 4) for k, v in feats.items()},
               **{f"snr_{k}": round(v, 1) for k, v in snr.items()},
               "diagnosis": verdict, "correct": verdict == label}
        results.append(row)
        print(f"{label:<18}{rpm:>6.0f}{feats['rms']:>8.3f}{feats['kurtosis']:>7.2f}"
              f"{snr['BPFI']:>10.1f}{snr['BPFO']:>10.1f}{snr['BSF']:>9.1f}  {verdict}")

    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    acc = sum(r["correct"] for r in results)
    print(f"\nCorrect diagnoses: {acc}/{len(results)}")
    ff = plot_data["Healthy"][4]
    print("Fault frequencies at %.0f rpm: " % results[0]["rpm"] +
          ", ".join(f"{k} = {v:.1f} Hz" for k, v in ff.items()))

    make_figures(plot_data, results)


def make_figures(plot_data, results):
    colors = {"Healthy": "#1b7f3b", "Inner Race Fault": "#c62828",
              "Ball Fault": "#6a1b9a", "Outer Race Fault": "#1565c0"}
    marker_style = {"BPFI": ("#c62828", "--"), "BPFO": ("#1565c0", "--"),
                    "BSF": ("#6a1b9a", ":")}

    # Fig 3: raw spectrum healthy vs inner race, BPFI marked
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for ax, label in zip(axes, ["Healthy", "Inner Race Fault"]):
        f, m, _, _, ff = plot_data[label]
        ax.plot(f, m, color=colors[label], lw=0.9, label=label)
        ax.axvline(ff["BPFI"], color="black", ls="--", lw=0.9)
        ax.text(ff["BPFI"] + 4, ax.get_ylim()[1] * 0.85,
                f"BPFI = {ff['BPFI']:.1f} Hz", fontsize=8)
        ax.set_ylabel("Amplitude (g)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3, ls="--")
    axes[1].set_xlim(0, 600)
    axes[1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Hann-windowed FFT spectrum (drive-end, 1797 rpm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_spectrum_comparison.png"), dpi=200)
    plt.close(fig)

    # Fig 4: envelope spectra for all four conditions
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
    for ax, (label, (_, _, f, m, ff)), r in zip(axes, plot_data.items(), results):
        ax.plot(f, m, color=colors[label], lw=0.9)
        for name, f0 in ff.items():
            c, ls = marker_style[name]
            ax.axvline(f0, color=c, ls=ls, lw=0.8, alpha=0.8)
        ax.set_title(f"{label}  ->  diagnosed: {r['diagnosis']}",
                     fontsize=9, loc="left")
        ax.set_ylabel("Env. amp.")
        ax.grid(alpha=0.3, ls="--")
    handles = [plt.Line2D([], [], color=c, ls=ls, label=n)
               for n, (c, ls) in marker_style.items()]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlim(0, 400)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Envelope spectra (band-pass 2-5 kHz + Hilbert transform)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_envelope_spectra.png"), dpi=200)
    plt.close(fig)

    # Fig 5: time-domain features
    labels = [r["recording"] for r in results]
    short = ["Healthy", "Inner race", "Ball", "Outer race"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, key, title in zip(axes, ["rms", "crest", "kurtosis"],
                              ["RMS (g)", "Crest factor", "Kurtosis"]):
        ax.bar(short, [r[key] for r in results],
               color=[colors[l] for l in labels])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.grid(axis="y", alpha=0.3, ls="--")
    axes[2].axhline(3, color="gray", ls="--", lw=0.8)
    axes[2].text(3.4, 3.2, "Gaussian = 3", fontsize=7, ha="right", color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_time_features.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
