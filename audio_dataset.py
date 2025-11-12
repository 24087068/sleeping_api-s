import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import librosa
from scipy.signal import find_peaks

# EDA:
def audio_eda(audio_path):
    # Info for all audio files
    audio_data = [
        {"firearm": f, "filename": a, "path": os.path.join(audio_path, f, a)}
        for f in os.listdir(audio_path)
        if os.path.isdir(os.path.join(audio_path, f))
        for a in os.listdir(os.path.join(audio_path, f))
    ]
    audio_df = pd.DataFrame(audio_data)
    counts = audio_df['firearm'].value_counts()
    dur = []
    sfreq = []
    peak_freqs = []

    # GenAi, for loop fix for peaks: https://chatgpt.com/share/6914546f-d378-800a-a6f5-251d076f6149
    for i in audio_df.index:
        p = audio_df.at[i, "path"]
        y, sr = librosa.load(p)
        N = len(y)
        freqs = fftfreq(N, 1 / sr)
        fft_values = np.abs(fft(y))
        positive = freqs > 0
        peak = freqs[positive][np.argmax(fft_values[positive])]
        dur.append(N / sr)
        sfreq.append(sr)
        peak_freqs.append(peak)

    # Basic statistics:
    print("Audio files:", len(audio_df))
    print("Unique firearms:", audio_df['firearm'].nunique())
    print("Audios per firearm:", counts.to_dict())
    print("Duration (s):", np.min(dur), "-", np.max(dur), ", mean:", np.mean(dur), 2)
    print("Sfreq (Hz):", set(sfreq))
    print("Peak frequencies (Hz):", np.min(peak_freqs), "-", np.max(peak_freqs), ",mean:", np.mean(peak_freqs))

    # Plot signals (based on first audio file):
    for firearm in audio_df['firearm'].unique():
        first_path = os.path.join(audio_path, firearm, audio_df[audio_df['firearm'] == firearm]['filename'].iloc[0])
        y, sr = librosa.load(first_path)
        N = len(y)
        t = np.arange(len(y)) / sr
        y_abs = np.abs(y)
        # GenAi, Detecting peaks: https://chatgpt.com/share/6914c756-9450-800a-9dce-4b2c992223bb
        y_env = np.convolve(y_abs, np.ones(2000) / 2000, mode='same') # smoothing variable, 25 fitted the best for spotting gunshots
        amp_peaks, _ = find_peaks(
            y_env,
            prominence=0.25 * np.max(y_env), # sensitivity variable, 25 fitted the best for spotting gunshots
            distance=int(0.0002 * sr) # interval variable, 0.0005 fitted the best for spotting gunshots
        )
        plt.figure(figsize=(10, 1))
        plt.plot(t, y)
        plt.scatter(t[amp_peaks], y[amp_peaks], color="red")
        plt.title(f"{firearm} Signal Met Mapped Gunshots")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

        # FFT magnitude
        freqs = fftfreq(N, 1 / sr)
        fft_values = np.abs(fft(y))
        positive = freqs > 0
        plt.figure(figsize=(10, 1))
        plt.plot(freqs[positive], fft_values[positive])
        plt.title(f"{firearm} FFT Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()

    return audio_df, counts, dur, sfreq, peak_freqs


# PREPROCESSING:
def extract_avg_features(audio_path):
    """
    Preprocessing: extracts average shots per second (how fast firearm fires) and average peak amplitude per firearm (how lod it is).
    shots_per_sec are scaled to 1s.
    """
    rows = []
    for firearm in os.listdir(audio_path):
        firearm_dir = os.path.join(audio_path, firearm)
        shots_list = []
        peak_list = []
        for fname in os.listdir(firearm_dir):
            path = os.path.join(firearm_dir, fname)
            y, sr = librosa.load(path, sr=None, mono=True)
            y_env = np.convolve(np.abs(y), np.ones(2000)/2000, mode='same') # smoothing variable, 25 fitted the best for spotting gunshots
            amp_peaks, _ = find_peaks(
                y_env,
                prominence=0.25 * np.max(y_env), # sensitivity variable, 25 fitted the best for spotting gunshots
                distance=int(0.0001 * sr) # interval variable, 0.0005 fitted the best for spotting gunshots
            )
            shots = len(amp_peaks)
            shots_per_sec = shots / 0.5
            peak_amp = np.max(np.abs(y))
            shots_list.append(shots_per_sec)
            peak_list.append(peak_amp)

        rows.append({
            "firearm": firearm,
            "avg_shots_per_sec": np.mean(shots_list),
            "avg_peak_amplitude": np.mean(peak_list)
        })
    df = pd.DataFrame(rows)
    return df