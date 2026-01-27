import pandas as pd
from typing import Optional


def apply_cleaning(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_numeric: Optional[str] = None,       # "median" | "mean" | None
    fill_categorical: Optional[str] = None,   # "mode" | "unknown" | None
) -> pd.DataFrame:

    df = df.copy()  # avoid mutating the original dataset

    # 1) Remove duplicate rows if enabled
    if drop_duplicates:
        df = df.drop_duplicates()

    # 2) Fill missing numeric values (median or mean) if selected
    if fill_numeric in {"median", "mean"}:
        num_cols = df.select_dtypes(include="number").columns

        for c in num_cols:
            # Only compute fill value if this column has missing values
            if df[c].isna().any():
                val = df[c].median() if fill_numeric == "median" else df[c].mean()
                df[c] = df[c].fillna(val)

    # 3) Fill missing categorical values (mode or "Unknown") if selected
    if fill_categorical in {"mode", "unknown"}:
        obj_cols = df.select_dtypes(include=["object", "string"]).columns

        for c in obj_cols:
            if df[c].isna().any():
                if fill_categorical == "mode":
                    # mode() returns possibly multiple values; choose the first
                    mode = df[c].mode(dropna=True)
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                else:
                    fill_val = "Unknown"

                df[c] = df[c].fillna(fill_val)

    # Reset index after any row drops to keep DataFrame clean
    return df.reset_index(drop=True)
