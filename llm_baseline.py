# llm_baseline.py
# Универсальный LLM-пайплайн для задач релевантности
# Weighted F1, SentenceTransformer, CatBoost

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

from llm_utils import auto_find_file, load_file, merge_items, make_pair_text


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VAL_SIZE = 0.2
RANDOM_STATE = 42


def main():

    # === 1. Load files ===
    items_path = auto_find_file("items")
    train_path = auto_find_file("train")
    test_path  = auto_find_file("test")

    items = load_file(items_path)
    train = load_file(train_path)
    test  = load_file(test_path)

    # === 2. Merge left/right ===
    train = merge_items(train, items, side="left")
    train = merge_items(train, items, side="right")

    test = merge_items(test, items, side="left")
    test = merge_items(test, items, side="right")

    # === 3. Build pair text ===
    train["pair_text"] = make_pair_text(train)
    test["pair_text"]  = make_pair_text(test)

    # === 4. Encode labels ===
    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    # === 5. Embeddings ===
    model = SentenceTransformer(MODEL_NAME)

    X_emb = model.encode(train["pair_text"].tolist(), show_progress_bar=True)
    X_test_emb = model.encode(test["pair_text"].tolist(), show_progress_bar=True)

    # === 6. Train/val split ===
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_emb,
        y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # === 7. Train classifier ===
    clf = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=700,
        depth=8,
        learning_rate=0.05,
        random_seed=RANDOM_STATE,
        verbose=100
    )

    clf.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    # === 8. Validation score ===
    val_pred = clf.predict(X_val).reshape(-1)
    f1 = f1_score(y_val, val_pred, average="weighted")

    print("\nWeighted F1:", f1)
    print("Classes:", le.classes_)

    # === 9. Retrain full ===
    clf.fit(X_emb, y)

    # === 10. Predict test ===
    test_pred = clf.predict(X_test_emb).reshape(-1)
    labels = le.inverse_transform(test_pred)

    # === 11. Submission ===
    submission = pd.DataFrame({
        "pair_id": test["pair_id"],
        "label": labels
    })

    submission.to_csv("submission_llm.csv", index=False)
    print("\nSaved to submission_llm.csv")
    print(submission.head())


if name == "__main__":
    main()
