import os
import argparse
import pandas as pd
import numpy as np
from src.extract_acoustic_features import extract_dataset_acoustic
from src.transcribe_whisper import transcribe_audio_files
from src.extract_linguistic_features import extract_dataset_linguistic
from src.fuse_features import fuse_multimodal_features
from src.train_and_evaluate import evaluate_models_cv, train_final_model

def main():
    parser = argparse.ArgumentParser(description="SHL Grammar Scoring Engine Runner")
    parser.add_argument("--data_dir", type=str, default="Dataset_Final", help="Path to Dataset_Final")
    parser.add_argument("--output_sub", type=str, default="final_submission.csv", help="Path for output submission")
    args = parser.parse_args()

    train_csv = os.path.join(args.data_dir, "train.csv")
    test_csv = os.path.join(args.data_dir, "test.csv")
    train_audio = os.path.join(args.data_dir, "train")
    test_audio = os.path.join(args.data_dir, "test")

    os.makedirs("artifacts", exist_ok=True)
    ac_train = "artifacts/features_acoustic_train.csv"
    ac_test = "artifacts/features_acoustic_test.csv"
    extract_dataset_acoustic(train_csv, train_audio, ac_train, is_train=True)
    extract_dataset_acoustic(test_csv, test_audio, ac_test, is_train=False)

    trans_train_cache = "artifacts/transcripts_train.json"
    trans_test_cache = "artifacts/transcripts_test.json"
    t_train = transcribe_audio_files(train_csv, train_audio, trans_train_cache, model_size="tiny")
    t_test = transcribe_audio_files(test_csv, test_audio, trans_test_cache, model_size="tiny")

    ling_train = "artifacts/features_linguistic_train.csv"
    ling_test = "artifacts/features_linguistic_test.csv"
    extract_dataset_linguistic(ac_train, t_train, ling_train, is_train=True)
    extract_dataset_linguistic(ac_test, t_test, ling_test, is_train=False)

    comb_train = "artifacts/features_combined_train.csv"
    comb_test = "artifacts/features_combined_test.csv"
    fuse_multimodal_features(ac_train, ling_train, comb_train, is_train=True)
    df_test = fuse_multimodal_features(ac_test, ling_test, comb_test, is_train=False)

    print("\nEvaluating 5-Fold Cross Validation...")
    evaluate_models_cv(comb_train)

    print("\nTraining Final Model...")
    model, feat_cols, train_rmse = train_final_model(comb_train)

    print("\nGenerating Test Predictions...")
    test_preds = np.clip(model.predict(df_test[feat_cols].values), 0.0, 5.0)

    sub = pd.DataFrame({"filename": df_test["filename"], "label": test_preds})
    sub.to_csv(args.output_sub, index=False)
    print(f"Submission generated successfully at: {args.output_sub}")

if __name__ == "__main__":
    main()
