# llm_utils.py
import os
import pandas as pd


def auto_find_file(pattern):
    """Автоматически ищет файл по части имени: train / test / items"""
    for f in os.listdir("."):
        low = f.lower()
        if pattern in low and (low.endswith(".csv") or low.endswith(".parquet")):
            return f
    raise FileNotFoundError(pattern)


def load_file(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Unknown file format")


def merge_items(df, items, side="left"):
    """Добавляет title и text для left/right id"""
    rename_map = {
        "item_id": f"{side}_id",
        "title": f"title_{side}",
        "text": f"text_{side}",
    }
    return df.merge(items.rename(columns=rename_map), on=f"{side}_id", how="left")


def make_pair_text(df):
    """Создаёт текст пары в формате:
       title_left + text_left + [SEP] + title_right + text_right
    """
    return (
        df["title_left"].fillna("") + " " +
        df["text_left"].fillna("")  + " [SEP] " +
        df["title_right"].fillna("") + " " +
        df["text_right"].fillna("")
    ).astype(str)
