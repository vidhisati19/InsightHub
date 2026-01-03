from __future__ import annotations
import pandas as pd
import plotly.express as px


def dist_plot(df: pd.DataFrame, col: str):
    """Distribution plot for a selected column (numeric histogram or categorical bar)."""
    s = df[col]

    if pd.api.types.is_numeric_dtype(s):
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution: {col}")
        fig.update_layout(bargap=0.05)
        return fig

    # dates: show counts over time by month
    if pd.api.types.is_datetime64_any_dtype(s):
        temp = df[[col]].dropna().copy()
        temp["month"] = temp[col].dt.to_period("M").astype(str)
        counts = temp["month"].value_counts().sort_index()
        fig = px.bar(x=counts.index, y=counts.values, title=f"Counts by month: {col}", labels={"x":"Month","y":"Count"})
        return fig
    
    if s.nunique(dropna=True) / max(len(s), 1) > 0.9:
        fig = px.bar(
            x=s.value_counts().head(20).index.astype(str),
            y=s.value_counts().head(20).values,
            title=f"Top values (ID-like): {col}"
        )
        return fig

    # categorical
    vc = s.fillna("Missing").value_counts(dropna=False).head(20)
    fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Top values: {col}", labels={"x":"Value","y":"Count"})
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def corr_heatmap(corr: pd.DataFrame):
    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        title="Correlation Heatmap (numeric columns)",
    )
    return fig