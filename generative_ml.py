# generative_ml_baseline.py
#
#
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

    pr
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
