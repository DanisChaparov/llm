
    """Создаёт текст пары в формате:
       title_left + text_left + [SEP] + title_right + text_right
    """
    return (
        df["title_left"].fillna("") + " " +
        df["text_left"].fillna("")  + " [SEP] " +
        df["title_right"].fillna("") + " " +
        df["text_right"].fillna("")
    ).astype(str)
