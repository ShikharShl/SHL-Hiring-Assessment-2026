"""Mean/std-pooled Whisper-encoder embeddings (mid + last layer) over the first 60 s of each clip."""
import sys, glob, os, numpy as np, soundfile as sf, torch
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
from transformers import WhisperFeatureExtractor, WhisperModel

def main(out, name="openai/whisper-base"):
    torch.set_num_threads(4)
    fe = WhisperFeatureExtractor.from_pretrained(name)
    enc = WhisperModel.from_pretrained(name).encoder.eval()
    files = sorted(glob.glob("C:/dataset/audios_train/*.wav")) + sorted(glob.glob("C:/dataset/audios_test/*.wav"))
    names, rows = [], []
    for i, f in enumerate(files):
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        chunks = [y[s:s + 480000] for s in (0, 480000) if len(y[s:s + 480000]) > 16000] or [y[:480000]]
        feats = []
        for c in chunks[:2]:
            x = fe(c, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                hs = enc(x, output_hidden_states=True).hidden_states
            n = max(1, int(len(c) / 16000 * 50))  # 50 frames/s, drop padded frames
            for L in (len(hs) // 2, len(hs) - 1):
                h = hs[L][0, :n]
                feats.append(torch.cat([h.mean(0), h.std(0)]).numpy())
        d = len(feats[0])
        per_layer = [np.mean([feats[j] for j in range(k, len(feats), 2)], axis=0) for k in (0, 1)]
        rows.append(np.concatenate(per_layer)); names.append(os.path.basename(f))
        if (i + 1) % 25 == 0: print(i + 1, flush=True)
    np.save(out, np.array(rows)); open(out + ".names", "w").write("\n".join(names)); print("done", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
