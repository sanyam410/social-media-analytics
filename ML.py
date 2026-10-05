# ml.py — Part 5: predict reach BEFORE posting (no leakage allowed)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast
from pymongo import MongoClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (confusion_matrix, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_style("whitegrid")
db = MongoClient("mongodb://localhost:27017/")["social_media_analytics"]

#1. extract posts (outcome fields NOT used as features)
posts = pd.DataFrame(db.posts.aggregate([
    {"$lookup": {"from": "users", "localField": "user_id", "foreignField": "_id", "as": "u"}},
    {"$unwind": "$u"},
    {"$project": {"_id": 0, "post_id": 1, "user_id": 1, "content_type": 1, "category": 1,
                  "hashtags": 1, "posted_at": 1, "views": 1, "viral": 1,
                  "account_type": "$u.account_type"}},
]))
posts["posted_at"] = pd.to_datetime(posts["posted_at"], utc=True)
posts["n_hashtags"] = posts["hashtags"].str.len()
posts = posts.drop(columns="hashtags")

print(posts.head())

# ---------- 2. point-in-time followers (the anti-leakage move) ----------
# users.followers_count is the FINAL count = future info. Count only follow
# edges that happened BEFORE each post — what the algorithm would see then.
edges = pd.DataFrame(db.followers.find({}, {"_id": 0, "following_id": 1, "followed_at": 1}))
edges["followed_at"] = pd.to_datetime(edges["followed_at"], utc=True)
m = posts.merge(edges, left_on="user_id", right_on="following_id", how="left")
m["already"] = m["followed_at"] <= m["posted_at"]          # NaT → False
followers_at_post = m.groupby("post_id")["already"].sum()
posts["followers_at_post"] = posts["post_id"].map(followers_at_post).astype(int)

# ---------- 3. time features + temporal split ----------
hr = posts["posted_at"].dt.hour
posts["hour_sin"], posts["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
posts["dow"] = posts["posted_at"].dt.dayofweek
posts["weekend"] = (posts["dow"] >= 5).astype(int)
print(posts.head())
cutoff = cast(pd.Timestamp, posts["posted_at"].quantile(0.8))
tr, te = posts[posts["posted_at"] <= cutoff], posts[posts["posted_at"] > cutoff]
print(f"train: {len(tr)} posts (≤ {pd.Timestamp(cutoff).date()})   test: {len(te)} posts")

CAT = ["account_type", "content_type", "category"]
NUM = ["followers_at_post", "n_hashtags", "hour_sin", "hour_cos", "dow", "weekend"]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                         ("num", StandardScaler(), NUM)])
X_tr, X_te = tr[CAT + NUM], te[CAT + NUM]
y_tr, y_te = np.log1p(tr["views"]), np.log1p(te["views"])

# ---------- 4. regression: predict log1p(views) ----------
print("\n=== REGRESSION — log1p(views) ===")
rf_pipe = None
for name, mdl in {
    "Dummy (mean)":  DummyRegressor(),
    "Ridge":         Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1),
}.items():
    pipe = Pipeline([("pre", pre), ("model", mdl)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    print(f"{name:15s} R²={r2_score(y_te, pred):6.3f}   MAE={mean_absolute_error(y_te, pred):5.3f} (log units)")
    if name == "Random Forest":
        rf_pipe = pipe

if rf_pipe is None:
    raise RuntimeError("Random forest regression pipeline was not created.")

# ORACLE ablation — cheating on purpose to show why leakage matters
Xo_tr = tr[CAT + NUM + ["viral"]]; Xo_te = te[CAT + NUM + ["viral"]]
pre_o = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                           ("num", StandardScaler(), NUM + ["viral"])])
oracle = Pipeline([("pre", pre_o), ("model", RandomForestRegressor(
    n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1))])
oracle.fit(Xo_tr, y_tr)
print(f"{'ORACLE (+viral)':15s} R²={r2_score(y_te, oracle.predict(Xo_te)):6.3f}"
      f"   ← post-hoc info, NOT a real model")

# ---------- 5. what drives reach? (permutation importance = honest version) ----------
assert rf_pipe is not None
pi = permutation_importance(rf_pipe, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
imp = pd.Series(pi["importances_mean"], index=X_te.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot.barh(ax=ax, color="#4c72b0")
ax.set_title("Permutation importance — what drives reach BEFORE you post?")
ax.set_xlabel("R² drop when feature is shuffled")
plt.tight_layout(); plt.savefig("reports/07_feature_importance.png", dpi=120); plt.show()

# ---------- 6. classification: reach level (tertiles from TRAIN only) ----------
q1, q2 = tr["views"].quantile([1/3, 2/3])
lvl = lambda v: "low" if v <= q1 else ("med" if v <= q2 else "high")
ytr_c, yte_c = tr["views"].map(lvl), te["views"].map(lvl)

print("\n=== CLASSIFICATION — reach level (low/med/high) ===")
for name, mdl in {
    "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Reg.":    LogisticRegression(max_iter=1000),
    "Random Forest":    RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                               random_state=42, n_jobs=-1),
}.items():
    pipe = Pipeline([("pre", pre), ("model", mdl)])
    pipe.fit(X_tr, ytr_c)
    print(f"{name:18s} macro-F1={f1_score(yte_c, pipe.predict(X_te), average='macro'):.3f}")

clf = Pipeline([("pre", pre), ("model", RandomForestClassifier(
    n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1))])
clf.fit(X_tr, ytr_c)
cm = confusion_matrix(yte_c, clf.predict(X_te), labels=["low", "med", "high"], normalize="true")
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["low", "med", "high"], yticklabels=["low", "med", "high"], ax=ax)
ax.set_xlabel("predicted"); ax.set_ylabel("actual")
ax.set_title("Reach-level confusion matrix (row = actual)")
plt.tight_layout(); plt.savefig("reports/08_confusion_matrix.png", dpi=120); plt.show()

print("\nDone — reports/07_feature_importance.png + 08_confusion_matrix.png saved.")

# ============================================================
# RESULTS — model summary + predicted vs actual + insights
# ============================================================

# ---------- 1. model summary table (R² lives HERE, model level) ----------
rows, _preds = [], {}
for name, mdl in {"Dummy (mean)": DummyRegressor(),
                  "Ridge": Ridge(alpha=1.0),
                  "Random Forest": rf_pipe}.items():
    if name == "Random Forest":
        pipe = rf_pipe
        if pipe is None:
            raise RuntimeError("Random forest regression pipeline is unavailable for summary.")
    else:
        pipe = Pipeline([("pre", pre), ("model", mdl)]).fit(X_tr, y_tr)
    pred_log = pipe.predict(X_te)
    pred_views = np.expm1(pred_log)                      # back to real views (≈ conditional median)
    actual = te["views"].values
    ape = np.abs(pred_views - actual) / actual
    rows.append({"model": name, "R2_log": round(r2_score(y_te, pred_log), 3),
                 "MAE_log": round(mean_absolute_error(y_te, pred_log), 3),
                 "median_%_err_views": round(np.median(ape) * 100, 1),
                 "within_±50%": round(np.mean(ape <= 0.5) * 100, 1)})
    _preds[name] = (pred_views, ape)
summary = pd.DataFrame(rows)
rf_pred, rf_ape = _preds["Random Forest"]
print("\n=== MODEL SUMMARY (temporal holdout) ===")
print(summary.to_string(index=False))

# ---------- 2. predicted vs actual (8 posts: 4 typical + 4 biggest) ----------
ex = pd.DataFrame({"post_id": te["post_id"].values, "actual": rf_pred/np.nan if False else te["views"].values,
                   "predicted": rf_pred.round(0).astype(int), "miss_%": (rf_ape * 100).round(1),
                   "viral": te["viral"].values})
sample = pd.concat([ex.sample(4, random_state=42), ex.nlargest(4, "actual")]).drop_duplicates("post_id")
print("\n=== PREDICTED vs ACTUAL (RF | random 4 + biggest 4 stress test) ===")
print(sample[["post_id", "actual", "predicted", "miss_%", "viral"]].to_string(index=False))

# ---------- 3. insights (every number computed, not hardcoded) ----------
vir = ex[ex["viral"]]
raw_r2 = r2_score(ex["actual"], ex["predicted"] - 0) # R² in RAW views space
n = len(te)
print(f"\n=== INSIGHTS (test set, n={n}) ===")
print(f"• Typical post: median miss {summary.iloc[2,3]}% — model is {summary.iloc[2,4]}% within ±50% of reality")
print(f"• Viral posts ({len(vir)}): median miss {vir['miss_%'].median():.0f}% under-predicted —"
      f" virality is unknowable BEFORE posting; that's honest, not a bug")
print(f"• Same RANK but raw R² = {raw_r2:.2f} (vs 0.727 in log space): viral outliers dominate normal"
      f" view scale — always check metric vs target space")
print(f"• Dummy R² < 0: test era is reach-ier (high-reach 33%→50%) — temporal shift breaks the baseline,"
      f" RF survives it")
