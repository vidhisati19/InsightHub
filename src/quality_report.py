import pandas as pd


def quality_summary(df: pd.DataFrame) -> dict:
    """High-level dataset quality metrics."""
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness report."""
    rep = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": df.isna().sum().values,
        "missing_%": (df.isna().mean().values * 100).round(2),
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    }).sort_values("missing", ascending=False, kind="stable")
    return rep.reset_index(drop=True)


def type_issues_report(df: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """
    Heuristics for common type issues:
    - numeric-like strings
    - date-like strings
    """
    issues = []

    for col in df.columns:
        if df[col].dtype != "object":
            continue

        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            continue

        s = s.head(sample_size)

        # numeric-like?
        numeric_parsed = pd.to_numeric(s, errors="coerce")
        numeric_rate = numeric_parsed.notna().mean()

        # date-like?
        date_parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        date_rate = date_parsed.notna().mean()

        issue = None
        suggestion = None

        if date_rate >= 0.7:
            issue = "Looks like dates stored as text"
            suggestion = "Convert to datetime"
        elif numeric_rate >= 0.7:
            issue = "Looks like numbers stored as text"
            suggestion = "Convert to numeric"

        if issue:
            issues.append({
                "column": col,
                "dtype_now": str(df[col].dtype),
                "issue": issue,
                "confidence_%": round(float(max(date_rate, numeric_rate) * 100), 1),
                "suggestion": suggestion,
            })

    return pd.DataFrame(issues)