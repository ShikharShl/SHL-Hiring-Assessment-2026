import pandas as pd
import numpy as np

def fuse_multimodal_features(acoustic_csv, linguistic_csv, output_csv, is_train=True):
    """
    Merges acoustic (45) and linguistic (18) feature sets on 'filename'.
    Ensures zero NaNs/Infs and exact column alignment.
    """
    df_ac = pd.read_csv(acoustic_csv)
    df_ling = pd.read_csv(linguistic_csv)

    if is_train and "label" in df_ling.columns:
        df_ling = df_ling.drop(columns=["label"])

    df_comb = pd.merge(df_ac, df_ling, on="filename")

    feat_cols = [c for c in df_comb.columns if c not in ["filename", "label"]]
    df_comb[feat_cols] = df_comb[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df_comb.to_csv(output_csv, index=False)
    print(f"Merged {len(df_comb)} instances with {len(feat_cols)} features -> {output_csv}")
    return df_comb
