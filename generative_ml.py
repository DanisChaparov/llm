# generative_ml_baseline.py
#
# Базовый ML-baseline для "генеративной" задачи:
# вместо генерации выбираем самый похожий train-образец
# и копируем его target_text.
#
# Ожидаемый формат:
#   train.csv: columns = ["input_text", "target_text"]
#   test.csv:  columns = ["input_text"]
#
# Выход:
#   submission_gen.csv: ["id", "target_text"]  (id можно поменять)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === НАСТРОЙКИ ===
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # или путь к локальной модели

INPUT_COL  = "input_text"   # переименуешь под задачу
TARGET_COL = "target_text"  # переименуешь под задачу
ID_COL     = "id"           # если в test есть id/pair_id и т.п.


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    # 1. Загрузка данных
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    # Проверим, что нужные колонки есть
    assert INPUT_COL in train.columns, f"{INPUT_COL} not in train columns"
    assert TARGET_COL in train.columns, f"{TARGET_COL} not in train columns"
    assert INPUT_COL in test.columns,  f"{INPUT_COL} not in test columns"

    if ID_COL not in test.columns:
        # если в тесте нет отдельного id, создадим его
        test[ID_COL] = np.arange(len(test))

    print("[INFO] Train shape:", train.shape)
    print("[INFO] Test shape :", test.shape)

    # 2. Модель эмбеддингов
    print("[INFO] Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # 3. Эмбеддинги train input_text
    print("[INFO] Encoding train input_text...")
    train_inputs = train[INPUT_COL].fillna("").astype(str).tolist()
    train_emb = model.encode(train_inputs, show_progress_bar=True, batch_size=64)
    train_emb = np.asarray(train_emb)

    # 4. Эмбеддинги test input_text
    print("[INFO] Encoding test input_text...")
    test_inputs = test[INPUT_COL].fillna("").astype(str).tolist()
    test_emb = model.encode(test_inputs, show_progress_bar=True, batch_size=64)
    test_emb = np.asarray(test_emb)

    # 5. Для каждого тестового объекта — ищем ближайший train
    print("[INFO] Searching nearest neighbors...")
    preds = []

    # Чтобы не считать косинус по формуле в цикле, можно заранее нормировать
    train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-9)
    test_norm  = test_emb  / (np.linalg.norm(test_emb,  axis=1, keepdims=True) + 1e-9)

    # По сути это kNN с k=1 по косинусу
    for i in tqdm(range(len(test_norm))):
        v = test_norm[i : i+1, :]               # (1, d)
        sims = (v @ train_norm.T)[0]            # косинусные похожести со всеми train
        best_idx = int(np.argmax(sims))         # индекс самого похожего
        best_target = train.iloc[best_idx][TARGET_COL]
        preds.append(best_target)

    # 6. Собираем сабмит
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: preds
    })

    out_path = "submission_gen.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print("[INFO] Saved:", out_path)
    print(submission.head())


if __name__ == "__main__":
    main()
