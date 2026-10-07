import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def evaluate_models_cv(train_csv, n_splits=5, random_state=42):
    """
    Runs 5-fold cross-validation across benchmark models on the combined feature set.
    """
    df = pd.read_csv(train_csv)
    feat_cols = [c for c in df.columns if c not in ["filename", "label"]]
    X = df[feat_cols].values
    y = df["label"].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = {
        "Combined_HistGBDT": HistGradientBoostingRegressor(random_state=random_state),
        "Combined_RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Combined_Ridge": RidgeCV(alphas=np.logspace(-3, 3, 20)),
    }

    results = {}
    for name, model in models.items():
        rmses, pearsons = [], []
        for train_idx, val_idx in kf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            r, _ = pearsonr(y_va, preds)
            pearsons.append(r if not np.isnan(r) else 0.0)

        results[name] = {
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "pearson_mean": np.mean(pearsons),
            "pearson_std": np.std(pearsons)
        }
        print(f"Model: {name:24s} | RMSE: {results[name]['rmse_mean']:.4f} +/- {results[name]['rmse_std']:.4f} | Pearson: {results[name]['pearson_mean']:.4f} +/- {results[name]['pearson_std']:.4f}")

    return results

def train_final_model(train_csv, n_estimators=100, random_state=42):
    """
    Trains the final selected Combined RandomForest model on all available training samples.
    """
    df = pd.read_csv(train_csv)
    feat_cols = [c for c in df.columns if c not in ["filename", "label"]]
    X = df[feat_cols].values
    y = df["label"].values

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    train_preds = model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    print(f"Final Model Trained on {len(X)} samples with {len(feat_cols)} features.")
    print(f"In-Sample Training RMSE: {train_rmse:.4f}")
    return model, feat_cols, train_rmse
