import pandas as pd
from typing import Optional

def apply_cleaning(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_numeric: Optional[str] = None,
    fill_categorical: Optional[str] = None,
) -> pd.DataFrame:

    df = df.copy()

    if drop_duplicates:
        df = df.drop_duplicates()

    # Numeric fill
    if fill_numeric in {"median", "mean"}:
        num_cols = df.select_dtypes(include="number").columns
        for c in num_cols:
            if df[c].isna().any():
                val = df[c].median() if fill_numeric == "median" else df[c].mean()
                df[c] = df[c].fillna(val)

    # Categorical fill
    if fill_categorical in {"mode", "unknown"}:
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for c in obj_cols:
            if df[c].isna().any():
                if fill_categorical == "mode":
                    mode = df[c].mode(dropna=True)
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                else:
                    fill_val = "Unknown"
                df[c] = df[c].fillna(fill_val)

    return df.reset_index(drop=True)