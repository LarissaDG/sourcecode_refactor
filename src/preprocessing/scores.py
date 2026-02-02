def add_score_columns(df, cols_to_compare):
    df = df.copy()

    df["Total Score"] = df[cols_to_compare].sum(axis=1)
    df["Avg Score"] = df["Total Score"] / len(cols_to_compare)

    return df
