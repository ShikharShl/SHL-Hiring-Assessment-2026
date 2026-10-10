"""v2 training: out-of-fold stacking of ridge(embeddings), SVR/LGBM(handcrafted+acoustic). Writes submission."""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import lightgbm as lgb

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

def main(work="C:/dataset_work", data="C:/dataset", out="final_submission_v2.csv", feats="C:/dataset_work/feats"):
    tr = pd.read_csv(f"{data}/train.csv"); te = pd.read_csv(f"{data}/test.csv")
    hc = pd.read_csv(f"{feats}/handcrafted.csv"); emb = np.load(f"{feats}/emb.npy")
    ac = pd.read_csv(f"{work}/acoustic.csv")
    idx = {f: i for i, f in enumerate(hc.filename)}
    tab = hc.merge(ac, on="filename").drop(columns=["filename"]).replace([np.inf, -np.inf], np.nan)
    tab = tab.fillna(tab.median()); tabv = tab.values
    pos = {f: i for i, f in enumerate(hc.filename)}
    ti = [pos[f] for f in tr.filename]; ei = [pos[f] for f in te.filename]
    y = tr.label.values.astype(float)
    wn = open(f"{feats}/wenc.npy.names").read().split("
"); W = np.load(f"{feats}/wenc.npy")
    W = W[[wn.index(f) for f in hc.filename]]  # align audio embeddings to the handcrafted-feature row order
    sets = {"wenc": W, "wtab": np.hstack([tabv, PCA(48, random_state=0).fit_transform(StandardScaler().fit_transform(W))]), "emb": emb, "tab": tabv, "all": np.hstack([tabv, PCA(32, random_state=0).fit_transform(emb)])}
    models = {
        "ridge_wenc": ("wenc", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 25)))),
        "svr_wenc": ("wenc", lambda: make_pipeline(StandardScaler(), PCA(64, random_state=0), SVR(C=3, epsilon=0.2))),
        "svr_wtab": ("wtab", lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.2))),
        "ridge_emb": ("emb", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 20)))),
        "svr_tab": ("tab", lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.2, gamma="scale"))),
        "svr_all": ("all", lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.2, gamma="scale"))),
        "lgb_all": ("all", lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1, n_jobs=4)),
    }
    oof = {k: np.zeros(len(y)) for k in models}; tp = {}
    for rep in range(2):
        for a, b in KFold(5, shuffle=True, random_state=rep).split(ti):
            for k, (s, mk) in models.items():
                X = sets[s][ti]; m = mk().fit(X[a], y[a]); oof[k][b] += m.predict(X[b]) / 2
    for k, (s, mk) in models.items():
        tp[k] = mk().fit(sets[s][ti], y).predict(sets[s][ei])
        print(f"{k:10s} CV RMSE {rmse(y, np.clip(oof[k],0,5)):.4f}  r={pearsonr(y, oof[k])[0]:.4f}")
    O = np.column_stack([oof[k] for k in models]); T = np.column_stack([tp[k] for k in models])
    # blend weights via non-negative ridge on OOF preds (nested-CV check of the blend itself)
    bl = Ridge(alpha=1.0, positive=True)
    nested = np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=99).split(O):
        nested[b] = Ridge(alpha=1.0, positive=True).fit(O[a], y[a]).predict(O[b])
    bl.fit(O, y)
    print("blend weights", dict(zip(models, bl.coef_.round(3))))
    print(f"BLEND CV RMSE {rmse(y, np.clip(nested,0,5)):.4f}  r={pearsonr(y, nested)[0]:.4f}")
    print(f"BLEND train RMSE (in-sample, final fit) {rmse(y, np.clip(bl.predict(O),0,5)):.4f}")
    pd.DataFrame({"filename": te.filename, "label": np.clip(bl.predict(T), 0, 5)}).to_csv(out, index=False)
    print("wrote", out)

if __name__ == "__main__":
    main()
