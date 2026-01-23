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
        t = np.arange(len(y)) / sr

        # RMS (Loudness) calculation for the loudness feature
        hop_v = 256 # hop variable, 256 worked best for precision
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_v)[0]
        times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_v)

        # GenAi, Detecting peaks: https://chatgpt.com/share/6914c756-9450-800a-9dce-4b2c992223bb
        amp_peaks, _ = find_peaks(
            rms,
            prominence=0.3 * np.max(rms), # prominence variable, 0.2 helps ignore the echo/noise
            height=0.5 * np.max(rms), # sensitivity variable, 0.1 fitted the best for spotting gunshots
            distance=int((0.05 * sr) / hop_v) # interval variable, 0.08 prevents double counting one shot
        )

        # Mapping peaks back to the main signal
        peak_indices = (times_rms[amp_peaks] * sr).astype(int)
        plt.figure(figsize=(10, 1))
        plt.plot(t, y, alpha=0.5)
        plt.scatter(t[peak_indices], y[peak_indices], color="red")
        plt.title(f"{firearm} Signal Met Mapped Gunshots")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

        # Loudness check visualization
        plt.figure(figsize=(10, 1))
        plt.plot(times_rms, rms, color='purple')
        plt.fill_between(times_rms, rms, color='purple', alpha=0.2)
        plt.title(f"{firearm} Loudness Feature")
        plt.xlabel("Time (s)")
        plt.ylabel("Loudness")
        plt.show()
    return audio_df, counts, dur, sfreq, peak_freqs


# PREPROCESSING:
def extract_avg_features(audio_path):
    """
    Preprocessing: extracts average shots per second and average peak amplitude per firearm (how loud it is).
    shots_per_sec are scaled to 1s.
    """
    rows = []
    for firearm in os.listdir(audio_path):
        firearm_dir = os.path.join(audio_path, firearm)
        if not os.path.isdir(firearm_dir):
            continue
        shots_list = []
        peak_list = []
        for fname in os.listdir(firearm_dir):
            path = os.path.join(firearm_dir, fname)
            y, sr = librosa.load(path, sr=None, mono=True)

            # RMS (Loudness) calculation for more stable feature extraction
            hop_v = 256 # hop variable, 256 worked best for precision
            rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_v)[0]
            amp_peaks, _ = find_peaks(
                rms,
                prominence=0.2 * np.max(rms), # prominence variable, 0.2 helps ignore the echo/noise
                height=0.1 * np.max(rms), # sensitivity variable, 0.1 fitted the best for spotting gunshots
                distance=int((0.08 * sr) / hop_v) # interval variable, 0.08 prevents double counting one shot
            )
            shots = len(amp_peaks)
            duration = len(y) / sr
            shots_per_sec = shots / duration

            # Using the max RMS value as the "Loudness" feature
            peak_loudness = np.max(rms)
            shots_list.append(shots_per_sec)
            peak_list.append(peak_loudness)
        rows.append({
            "firearm": firearm,
            "avg_shots_per_sec": np.mean(shots_list),
            "avg_loudness_amplitude": np.mean(peak_list)
        })
    df = pd.DataFrame(rows)
    return df