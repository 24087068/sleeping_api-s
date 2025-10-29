import os, random, numpy as np, pandas as pd, librosa
def audio_eda(audio_path):
    # Collect info for all audio files in subfolders
    audio_data = [
        {"firearm": f, "filename": a, "path": os.path.join(audio_path, f, a)}
        for f in os.listdir(audio_path)
        if os.path.isdir(os.path.join(audio_path, f))
        for a in os.listdir(os.path.join(audio_path, f))
    ]
    audio_df = pd.DataFrame(audio_data)
    counts = audio_df['firearm'].value_counts()
    # Lists for stats
    dur, srates, sizes = [], [], []
    # GenAI: fixing samples issues, https://chatgpt.com/share/6901dca5-7214-800a-aba6-5b43ba72e10b
    # Randomly sample some files (or use all if few)
    sample_indices = random.sample(range(len(audio_df)), min(10, len(audio_df)))
    for i in sample_indices:
        p = audio_df.at[i, "path"]
        y, sr = librosa.load(p, sr=None)
        dur.append(len(y) / sr)
        srates.append(sr)
        sizes.append(os.path.getsize(p) / 1024)  # in KB
    print("Audio files:", len(audio_df))
    print("Unique firearms:", audio_df['firearm'].nunique())
    print(f"Audios per firearm: {counts.to_dict()}")
    print("Size (KB) statistics:", np.min(sizes), np.max(sizes), np.mean(sizes), np.std(sizes))
    print("Duration (s) statistics:", np.min(dur), np.max(dur), np.mean(dur), np.std(dur))
    print("Sample rates (Hz):", sorted(set(srates)))
    return audio_df, counts, dur, srates, sizes