from __future__ import annotations
import pandas as pd
from pathlib import Path  # (not used currently; safe to remove if you want)
from datetime import datetime


def generate_eda_report(
    df: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    key_findings: list[str],
    outliers: pd.DataFrame,
    corr: pd.DataFrame,
) -> str:
    """Return an HTML string for the EDA report."""

    # Timestamp to show when the report was generated
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build a full HTML page as a single string
    # Uses pandas .to_html() to render tables quickly
    html = f"""
    <html>
    <head>
        <title>InsightHub — EDA Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; }}
            th {{ background-color: #f4f4f4; }}
            ul {{ margin-left: 20px; }}
        </style>
    </head>
    <body>
        <h1>InsightHub — EDA Report</h1>
        <p><b>Generated:</b> {ts}</p>
        <p><b>Rows:</b> {len(df)} | <b>Columns:</b> {df.shape[1]}</p>

        <h2>Key Findings</h2>
        <ul>
            {''.join(f"<li>{k}</li>" for k in key_findings)}
        </ul>

        <h2>Numeric Summary</h2>
        {numeric_summary.to_html(index=False)}

        <h2>Categorical Summary</h2>
        {categorical_summary.to_html(index=False)}

        <h2>Outlier Detection (IQR)</h2>
        {outliers.to_html(index=False)}

        <h2>Correlation Matrix</h2>
        {corr.round(3).to_html()}
    </body>
    </html>
    """

    return html
