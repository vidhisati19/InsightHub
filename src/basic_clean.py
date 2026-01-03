import pandas as pd


def drop_header_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sometimes CSVs contain an extra header row inside the data
    (e.g., a row containing 'Release Date'). This removes rows
    that exactly match the column names.
    """
    if df.empty:
        return df

    colnames = list(df.columns)

    # Keep rows that are NOT exactly equal to the column names
    mask = ~(df.astype(str).fillna("").apply(lambda r: list(r.values) == colnames, axis=1))
    return df.loc[mask].reset_index(drop=True)


def infer_and_cast_dates(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Tries to convert object columns to datetime if enough values parse successfully.
    threshold=0.7 means: if >=70% of non-null values parse -> treat as datetime.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].astype(str).str.strip()
            # Try parsing
            parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

            non_null = s.replace(["", "nan", "None"], pd.NA).dropna()
            if len(non_null) == 0:
                continue

            success_rate = parsed.notna().sum() / len(non_null)
            if success_rate >= threshold:
                df[col] = parsed

    return df


def basic_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_header_like_rows(df)
    df = infer_and_cast_dates(df)
    return df