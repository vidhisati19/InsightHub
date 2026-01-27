import pandas as pd


def quality_summary(df: pd.DataFrame) -> dict:
    """High-level dataset quality metrics."""
    return {
        # Basic shape info
        "rows": int(len(df)),
        "cols": int(df.shape[1]),

        # Missingness and duplicates
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),

        # Rough memory footprint (deep=True counts object/string memory too)
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness report."""
    rep = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],

        # Missing count and percent
        "missing": df.isna().sum().values,
        "missing_%": (df.isna().mean().values * 100).round(2),

        # Uniqueness is useful for spotting ID-like columns
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    }).sort_values("missing", ascending=False, kind="stable")

    return rep.reset_index(drop=True)


def type_issues_report(df: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """
    Heuristics for common type issues:
    - numeric-like strings (e.g., "123" stored as text)
    - date-like strings (e.g., "2020-01-01" stored as text)
    """
    issues = []

    for col in df.columns:
        # Only check object columns (most type problems happen here)
        if df[col].dtype != "object":
            continue

        # Work on non-null values, trimmed to avoid whitespace errors
        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            continue

        # Only sample first N values for speed (large datasets)
        s = s.head(sample_size)

        # Try parse as numeric
        numeric_parsed = pd.to_numeric(s, errors="coerce")
        numeric_rate = numeric_parsed.notna().mean()  # % that successfully parsed as numeric

        # Try parse as date
        date_parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        date_rate = date_parsed.notna().mean()        # % that successfully parsed as date

        issue = None
        suggestion = None

        # Prefer date detection if both look plausible
        if date_rate >= 0.7:
            issue = "Looks like dates stored as text"
            suggestion = "Convert to datetime"
        elif numeric_rate >= 0.7:
            issue = "Looks like numbers stored as text"
            suggestion = "Convert to numeric"

        # If we found a likely issue, record it
        if issue:
            issues.append({
                "column": col,
                "dtype_now": str(df[col].dtype),
                "issue": issue,
                "confidence_%": round(float(max(date_rate, numeric_rate) * 100), 1),
                "suggestion": suggestion,
            })

    return pd.DataFrame(issues)
