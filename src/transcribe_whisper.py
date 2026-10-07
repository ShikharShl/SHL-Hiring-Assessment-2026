import os
import json
import soundfile as sf
import numpy as np
import torch
import whisper

def transcribe_audio_files(csv_path, audio_dir, cache_json, model_size="tiny", num_threads=4):
    """
    Transcribes audio files using OpenAI Whisper (tiny) with CPU thread optimization
    and incremental JSON disk caching.
    """
    import pandas as pd
    torch.set_num_threads(num_threads)
    df = pd.read_csv(csv_path)

    transcripts = {}
    if os.path.exists(cache_json):
        with open(cache_json, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

    remaining = [f for f in df["filename"] if f not in transcripts]
    print(f"Transcribing {len(remaining)} remaining files with Whisper '{model_size}'...")

    if remaining:
        model = whisper.load_model(model_size)
        for idx, fname in enumerate(remaining):
            apath = os.path.join(audio_dir, fname)
            if os.path.exists(apath):
                try:
                    data, sr = sf.read(apath)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    res = model.transcribe(data.astype(np.float32), fp16=False, language="en")
                    transcripts[fname] = res.get("text", "").strip()
                except Exception as e:
                    print(f"Error transcribing {fname}: {e}")
                    transcripts[fname] = ""
            else:
                transcripts[fname] = ""

            if (idx + 1) % 10 == 0 or (idx + 1) == len(remaining):
                with open(cache_json, "w", encoding="utf-8") as f:
                    json.dump(transcripts, f, indent=2)
                print(f"  Cached {len(transcripts)}/{len(df)} transcripts...")

    print(f"Completed transcription for {len(transcripts)} files -> {cache_json}")
    return transcripts
