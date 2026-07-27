#
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
