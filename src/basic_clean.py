import pandas as pd


def drop_header_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sometimes CSVs contain an extra header row inside the data
    (e.g., a row containing 'Release Date'). This removes rows
    that exactly match the column names.
    """
    # If no data, nothing to fix
    if df.empty:
        return df

    # Store the actual header names
    colnames = list(df.columns)

    # Build a boolean mask:
    # True for rows that are NOT identical to the header row
    # Steps:
    # 1) df.astype(str): compare reliably
    # 2) fillna(""): avoid NaN comparison issues
    # 3) apply row-wise: check if row values == colnames exactly
    mask = ~(df.astype(str).fillna("").apply(lambda r: list(r.values) == colnames, axis=1))

    # Keep only valid rows and reset row index
    return df.loc[mask].reset_index(drop=True)


def infer_and_cast_dates(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Tries to convert object columns to datetime if enough values parse successfully.
    threshold=0.7 means: if >=70% of non-null values parse -> treat as datetime.
    """
    df = df.copy()  # avoid mutating the original DataFrame

    # Scan every column and try to convert only object/text columns
    for col in df.columns:
        if df[col].dtype == "object":
            # Strip whitespace to improve parse success
            s = df[col].astype(str).str.strip()

            # Attempt to parse dates; invalid values become NaT
            parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

            # Count only “real” values (ignore blanks / string-nan / None)
            non_null = s.replace(["", "nan", "None"], pd.NA).dropna()
            if len(non_null) == 0:
                continue

            # Success rate = (how many parsed) / (how many non-null values)
            success_rate = parsed.notna().sum() / len(non_null)

            # If enough values parse as dates, convert the whole column
            if success_rate >= threshold:
                df[col] = parsed

    return df


def basic_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Run lightweight prep steps in a clean pipeline order
    df = drop_header_like_rows(df)
    df = infer_and_cast_dates(df)
    return df
