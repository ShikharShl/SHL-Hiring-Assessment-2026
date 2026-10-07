import re
import math
import pandas as pd

FILLER_WORDS = {"um", "uh", "like", "ah", "er", "hmm", "you know", "well", "so", "basically", "actually"}

def compute_linguistic_metrics(text, duration):
    """
    Computes 18 robust, explainable linguistic and lexical statistics
    from an ASR transcript and corresponding audio duration.
    """
    text_clean = text.strip()
    words = [w.strip(".,!?;:\"'()[]{}").lower() for w in text_clean.split() if w.strip(".,!?;:\"'()[]{}")]

    word_count = len(words)
    character_count = sum(len(w) for w in words)

    sentences = [s.strip() for s in re.split(r'[.!?]+', text_clean) if s.strip()]
    sentence_count = max(1, len(sentences))

    avg_words_per_sentence = word_count / float(sentence_count)
    avg_word_length = character_count / float(max(1, word_count))

    unique_words = set(words)
    unique_word_count = len(unique_words)
    type_token_ratio = unique_word_count / float(max(1, word_count))
    vocabulary_diversity = unique_word_count / math.sqrt(max(1, word_count))

    repeated_words = 0
    for i in range(len(words) - 1):
        if words[i] == words[i+1]:
            repeated_words += 1
    repeated_word_ratio = repeated_words / float(max(1, word_count))

    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_word_rate = filler_count / float(max(1, word_count))

    words_per_second = word_count / float(max(0.1, duration))
    sentence_rate = sentence_count / float(max(0.1, duration))

    question_count = text_clean.count("?")
    exclamation_count = text_clean.count("!")

    long_words = sum(1 for w in words if len(w) > 6)
    short_words = sum(1 for w in words if len(w) <= 3)
    long_word_ratio = long_words / float(max(1, word_count))
    short_word_ratio = short_words / float(max(1, word_count))

    content_words = sum(1 for w in words if len(w) >= 4 and w not in FILLER_WORDS)
    lexical_density = content_words / float(max(1, word_count))

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_word_length": avg_word_length,
        "unique_word_count": unique_word_count,
        "type_token_ratio": type_token_ratio,
        "vocabulary_diversity": vocabulary_diversity,
        "repeated_word_ratio": repeated_word_ratio,
        "filler_word_count": filler_count,
        "filler_word_rate": filler_word_rate,
        "words_per_second": words_per_second,
        "sentence_rate": sentence_rate,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "long_word_ratio": long_word_ratio,
        "short_word_ratio": short_word_ratio,
        "lexical_density": lexical_density,
    }

def extract_dataset_linguistic(acoustic_csv, transcripts_dict, output_csv, is_train=True):
    df_ac = pd.read_csv(acoustic_csv)
    records = []
    for _, row in df_ac.iterrows():
        fname = row["filename"]
        text = transcripts_dict.get(fname, "")
        dur = row["duration"]
        rec = {"filename": fname}
        if is_train and "label" in row:
            rec["label"] = row["label"]
        rec.update(compute_linguistic_metrics(text, dur))
        records.append(rec)
    res_df = pd.DataFrame(records)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved {len(res_df)} linguistic feature records to {output_csv}")
    return res_df
