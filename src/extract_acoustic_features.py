import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

def extract_audio_features(audio_path, target_sr=16000):
    """
    Extracts 45 robust, duration-normalized acoustic & prosodic features
    from an audio file using 25ms framing, 10ms hop length, and librosa/soundfile.
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    duration = len(y) / float(sr)
    if duration < 0.1:
        y = np.zeros(int(target_sr * 0.1), dtype=np.float32)
        duration = 0.1

    hop_length = int(0.010 * sr)
    win_length = int(0.025 * sr)
    n_fft = max(512, 1 << (win_length - 1).bit_length())

    # 1. RMS Energy & Dynamics
    rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    energy_variability = float(rms_std / (rms_mean + 1e-8))

    # 2. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win_length, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 3. Silence and Pause Metrics (< 0.01 threshold)
    silence_thresh = 0.01
    is_silent = rms < silence_thresh
    silence_ratio = float(np.mean(is_silent))
    active_speech_ratio = float(1.0 - silence_ratio)
    speech_pause_ratio = float(active_speech_ratio / (silence_ratio + 1e-8))

    # Silent intervals >= 200ms (20 consecutive frames)
    min_pause_frames = 20
    silent_blocks = []
    curr_len = 0
    for s in is_silent:
        if s:
            curr_len += 1
        else:
            if curr_len >= min_pause_frames:
                silent_blocks.append(curr_len * 0.010)
            curr_len = 0
    if curr_len >= min_pause_frames:
        silent_blocks.append(curr_len * 0.010)

    pause_freq_per_min = float((len(silent_blocks) / (duration + 1e-8)) * 60.0)
    mean_pause_duration = float(np.mean(silent_blocks)) if silent_blocks else 0.0

    # 4. Spectral Descriptors
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    spec_cent_mean = float(np.mean(spec_cent))
    spec_cent_std = float(np.std(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    spec_bw_mean = float(np.mean(spec_bw))
    spec_bw_std = float(np.std(spec_bw))

    spec_rf = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    spec_rf_mean = float(np.mean(spec_rf))
    spec_rf_std = float(np.std(spec_rf))

    try:
        spec_ct = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spec_ct_mean = float(np.mean(spec_ct))
        spec_ct_std = float(np.std(spec_ct))
    except Exception:
        spec_ct_mean = 0.0
        spec_ct_std = 0.0

    # 5. 13 MFCC Coefficients (Mean & Std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfcc_feats = {}
    for i in range(13):
        mfcc_feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        mfcc_feats[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))

    feats = {
        "duration": duration,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "energy_variability": energy_variability,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "silence_ratio": silence_ratio,
        "active_speech_ratio": active_speech_ratio,
        "speech_pause_ratio": speech_pause_ratio,
        "pause_freq_per_min": pause_freq_per_min,
        "mean_pause_duration": mean_pause_duration,
        "spec_cent_mean": spec_cent_mean,
        "spec_cent_std": spec_cent_std,
        "spec_bw_mean": spec_bw_mean,
        "spec_bw_std": spec_bw_std,
        "spec_rf_mean": spec_rf_mean,
        "spec_rf_std": spec_rf_std,
        "spec_ct_mean": spec_ct_mean,
        "spec_ct_std": spec_ct_std,
    }
    feats.update(mfcc_feats)
    return feats

def extract_dataset_acoustic(csv_path, audio_dir, output_csv, is_train=True):
    df = pd.read_csv(csv_path)
    records = []
    print(f"Extracting acoustic features for {len(df)} files from {audio_dir}...")
    for idx, row in df.iterrows():
        fname = row["filename"]
        apath = os.path.join(audio_dir, fname)
        feats = extract_audio_features(apath)
        if feats is not None:
            rec = {"filename": fname}
            if is_train and "label" in row:
                rec["label"] = row["label"]
            rec.update(feats)
            records.append(rec)
    res_df = pd.DataFrame(records)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved {len(res_df)} acoustic feature records to {output_csv}")
    return res_df
