import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier

# -------------------------------------
# UNIVERSAL LOADER (CSV or Parquet)
# -------------------------------------
def load(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Unsupported format:", path)

# -------------------------------------
# LOAD DATA
# -------------------------------------
items = load("items.csv")
train = load("train.csv")
test  = load("test.csv")

# -------------------------------------
# MERGE TEXTS
# -------------------------------------
items = items[["item_id", "title", "text"]]

train = train.merge(items, left_on="left_id", right_on="item_id", how="left")
train = train.merge(items, left_on="right_id", right_on="item_id", how="left",
                    suffixes=("_left", "_right"))

test = test.merge(items, left_on="left_id", right_on="item_id", how="left")
test = test.merge(items, left_on="right_id", right_on="item_id", how="left",
                  suffixes=("_left", "_right"))

# Combine title+text
train["text_left_full"]  = train["title_left"]  + ". " + train["text_left"]
train["text_right_full"] = train["title_right"] + ". " + train["text_right"]

test["text_left_full"]  = test["title_left"]  + ". " + test["text_left"]
test["text_right_full"] = test["title_right"] + ". " + test["text_right"]

# -------------------------------------
# LOAD EMBEDDING MODEL
# (must be in /shared/models on the cluster)
# -------------------------------------
model_emb = SentenceTransformer(
    "/shared/models/sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------
# MAKE EMBEDDINGS
# -------------------------------------
def build_features(df):
    left_emb = model_emb.encode(df["text_left_full"].tolist(), show_progress_bar=True)
    right_emb = model_emb.encode(df["text_right_full"].tolist(), show_progress_bar=True)

    # Combine embeddings
    return np.hstack([
        left_emb,
        right_emb,
        np.abs(left_emb - right_emb)  # distance
    ])

X = build_features(train)
X_test = build_features(test)

# -------------------------------------
# LABEL ENCODING
# -------------------------------------
label_map = {
    "no_relevant": 0,
    "relevant_minus": 1,
    "relevant": 2,
    "relevant_plus": 3
}

y = train["label"].map(label_map)

# -------------------------------------
# TRAIN CATBOOST
# -------------------------------------
model = CatBoostClassifier(
    iterations=600,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    verbose=100
)

model.fit(X, y)

# -------------------------------------
# PREDICT
# -------------------------------------
preds = model.predict(X_test).astype(int).flatten()

inv_map = {v: k for k, v in label_map.items()}
test["label"] = [inv_map[x] for x in preds]

# -------------------------------------
# SAVE SUBMISSION
# -------------------------------------
submission = test[["pair_id", "label"]]
submission.to_csv("submission.csv", index=False)

print("DONE! Saved submission.csv")
