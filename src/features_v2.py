"""v2 features: ASR confidence/timing, GPT-2 fluency, sentence embeddings, acoustic stats."""
import re, json, math, sys, os
import numpy as np, pandas as pd
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass

FILLERS = {"um", "uh", "ah", "er", "hmm", "erm", "like"}

def asr_features(rec):
    words = rec["words"]; text = rec["text"]
    toks = [re.sub(r"[^a-z']", "", w[0].lower()) for w in words]
    toks = [t for t in toks if t]
    n = len(toks)
    probs = np.array([w[3] for w in words]) if words else np.zeros(1)
    starts = np.array([w[1] for w in words]) if words else np.zeros(1)
    ends = np.array([w[2] for w in words]) if words else np.zeros(1)
    gaps = starts[1:] - ends[:-1] if len(words) > 1 else np.zeros(1)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    slen = [len(s.split()) for s in sents] or [0]
    speech_time = max(ends.max() - starts.min(), 1e-3) if words else 1e-3
    f = {
        "n_words": n, "n_unique": len(set(toks)), "ttr": len(set(toks)) / max(n, 1),
        "root_ttr": len(set(toks)) / math.sqrt(max(n, 1)),
        "wpm": n / speech_time * 60, "speech_span": speech_time,
        "p_mean": probs.mean(), "p_min": probs.min(), "p_std": probs.std(),
        "p_low30": (probs < .3).mean(), "p_low60": (probs < .6).mean(), "p_hi90": (probs > .9).mean(),
        "gap_mean": gaps.mean(), "gap_max": gaps.max(), "gap_n03": (gaps > .3).sum(), "gap_n1": (gaps > 1).sum(),
        "gap_frac": np.clip(gaps, 0, None).sum() / speech_time,
        "n_sent": len(sents), "sent_mean": np.mean(slen), "sent_max": np.max(slen), "sent_std": np.std(slen),
        "filler_rate": sum(t in FILLERS for t in toks) / max(n, 1),
        "rep_rate": sum(a == b for a, b in zip(toks, toks[1:])) / max(n, 1),
        "avg_wlen": np.mean([len(t) for t in toks]) if toks else 0,
        "long_rate": np.mean([len(t) > 6 for t in toks]) if toks else 0,
        "comma_rate": text.count(",") / max(n, 1),
        "first_start": starts.min() if words else 60, "last_end": ends.max() if words else 0,
    }
    return f

def gpt2_features(texts, model_name="gpt2"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_name); m = AutoModelForCausalLM.from_pretrained(model_name).eval()
    out = []
    for t in texts:
        ids = tok(tok.bos_token + (t or "."), return_tensors="pt", truncation=True, max_length=512)["input_ids"]
        with torch.no_grad():
            lg = m(ids).logits[0, :-1]
        nll = torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="none").numpy()
        out.append({"lm_mean": nll.mean(), "lm_med": np.median(nll), "lm_p90": np.percentile(nll, 90),
                    "lm_hi": (nll > 6).mean(), "lm_std": nll.std(), "lm_max": nll.max()})
    return pd.DataFrame(out)

def sent_embed(texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name).encode(texts, batch_size=16, show_progress_bar=False)

def build(asr_json, out_dir):
    d = json.load(open(asr_json)); os.makedirs(out_dir, exist_ok=True)
    names = sorted(d); texts = [d[k]["text"] for k in names]
    A = pd.DataFrame([asr_features(d[k]) for k in names]); A.insert(0, "filename", names)
    L = gpt2_features(texts)
    pd.concat([A, L], axis=1).to_csv(f"{out_dir}/handcrafted.csv", index=False)
    np.save(f"{out_dir}/emb.npy", sent_embed(texts))
    print("built", len(names))

if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
