from __future__ import annotations
import pandas as pd
import numpy as np


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()

    summ = pd.DataFrame({
        "column": num.columns,
        "count": num.count().values,
        "missing": num.isna().sum().values,
        "mean": num.mean(numeric_only=True).values,
        "median": num.median(numeric_only=True).values,
        "std": num.std(numeric_only=True).values,
        "min": num.min(numeric_only=True).values,
        "max": num.max(numeric_only=True).values,
    })
    return summ.sort_values("missing", ascending=False, kind="stable").reset_index(drop=True)


def categorical_summary(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    cat = df.select_dtypes(include=["object", "string", "category", "bool"])
    if cat.empty:
        return pd.DataFrame()

    rows = []
    for c in cat.columns:
        s = df[c].fillna("Missing")
        vc = s.value_counts(dropna=False).head(top_k)
        top_values = ", ".join([f"{idx} ({int(v)})" for idx, v in vc.items()][:5])
        rows.append({
            "column": c,
            "missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True)),
            "top_values": top_values,
        })
    return pd.DataFrame(rows).sort_values("missing", ascending=False, kind="stable").reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(numeric_only=True)


def iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-numeric-column outlier counts using IQR rule."""
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()

    rows = []
    for c in num.columns:
        s = num[c].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            out_count = 0
            lower = np.nan
            upper = np.nan
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_count = int(((s < lower) | (s > upper)).sum())

        rows.append({
            "column": c,
            "outliers": out_count,
            "outlier_%": round(out_count / max(len(s), 1) * 100, 2),
            "lower": lower,
            "upper": upper,
        })

    return pd.DataFrame(rows).sort_values("outliers", ascending=False, kind="stable").reset_index(drop=True)


def key_findings(df: pd.DataFrame, corr_top_k: int = 5) -> list[str]:
    """Simple rule-based bullets to generate 'wow' insights."""
    bullets: list[str] = []

    # Missingness
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    if len(miss_pct) > 0 and miss_pct.iloc[0] >= 20:
        top = miss_pct.head(3)
        bullets.append(
            "High missingness detected: " +
            ", ".join([f"{c} ({p:.1f}%)" for c, p in top.items()])
        )

    # Duplicates
    dups = int(df.duplicated().sum())
    if dups > 0:
        bullets.append(f"{dups:,} duplicate rows found (consider dropping duplicates).")

    # Outliers
    out = iqr_outliers(df)
    if not out.empty and out["outliers"].max() > 0:
        top = out.head(3)
        bullets.append(
            "Outliers detected (IQR rule): " +
            ", ".join([f"{r['column']} ({int(r['outliers'])})" for _, r in top.iterrows()])
        )

    # Correlations
    corr = correlation_matrix(df)
    if not corr.empty:
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    pairs.append((cols[i], cols[j], float(val)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        strong = [(a, b, v) for a, b, v in pairs if abs(v) >= 0.7]
        if strong:
            bullets.append(
                "Strong correlations (|r| ≥ 0.7): " +
                ", ".join([f"{a}–{b} (r={v:.2f})" for a, b, v in strong[:corr_top_k]])
            )

    if not bullets:
        bullets.append("No major red flags found. Dataset looks fairly clean for quick analysis.")

    return bullets