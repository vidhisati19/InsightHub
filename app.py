# --------------------------------------------
# app.py — Streamlit entrypoint for InsightHub
# --------------------------------------------

import streamlit as st              # Streamlit UI framework (widgets, layout, app state)
import pandas as pd                 # DataFrame operations + type checks

# ---- Import internal modules (project structure) ----
from src.basic_clean import basic_prepare
from src.data_load import load_csv
from src.quality_report import quality_summary, missing_report, type_issues_report
from src.cleaning_actions import apply_cleaning
from src.eda import numeric_summary, categorical_summary, correlation_matrix, iqr_outliers, key_findings
from src.viz import dist_plot, corr_heatmap
from src.modeling import train_baseline
from src.reporting import generate_eda_report


# ============================================================
# 1) STREAMLIT APP CONFIG (runs once when app starts / reloads)
# ============================================================
st.set_page_config(
    page_title="InsightHub",     # Browser tab title
    page_icon="📊",              # Favicon emoji
    layout="wide",               # Uses full page width
)

# App header
st.title("📊 InsightHub")
st.caption("Upload a CSV to instantly preview and understand your dataset.")


# ============================================================
# 2) SIDEBAR NAVIGATION (controls which page is shown)
# ============================================================
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Cleaning Report", "EDA Dashboard", "Modeling"],
    index=0
)
st.sidebar.divider()


# ============================================================
# 3) FILE UPLOAD (shared by ALL pages)
# ============================================================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# If user hasn’t uploaded anything yet, stop the app early
if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()


# ============================================================
# 4) LOAD DATA + BASIC PREP (shared by ALL pages)
# ============================================================
df = load_csv(uploaded_file)      # Convert uploaded file into a pandas DataFrame
df = basic_prepare(df)            # Light preprocessing (trim col names, parse dates, etc.)


# ============================================================
# 5) SIDEBAR CLEANING OPTIONS (shared by ALL pages)
#    These controls can modify df if user toggles "Apply cleaning"
# ============================================================
st.sidebar.subheader("Cleaning Options")

# Option 1: remove duplicate rows
drop_dupes = st.sidebar.checkbox("Drop duplicate rows", value=True)

# Option 2: fill numeric missing values
fill_num = st.sidebar.selectbox(
    "Fill missing numeric values",
    options=["None", "Median", "Mean"],
    index=0
)

# Option 3: fill categorical missing values
fill_cat = st.sidebar.selectbox(
    "Fill missing categorical values",
    options=["None", "Mode", "Unknown"],
    index=0
)

# Master toggle to actually apply cleaning changes to df
apply_opts = st.sidebar.checkbox("Apply cleaning to dataset", value=False)

# If enabled, transform df using your cleaning pipeline
if apply_opts:
    df = apply_cleaning(
        df,
        drop_duplicates=drop_dupes,
        # If "None", pass None, otherwise pass "median"/"mean"
        fill_numeric=None if fill_num == "None" else fill_num.lower(),
        # If "None", pass None, otherwise pass "mode"/"unknown"
        fill_categorical=None if fill_cat == "None" else fill_cat.lower(),
    )


# ============================================================
# PAGE 1: OVERVIEW
# Purpose: quick dataset summary + preview + column lookups
# ============================================================
if page == "Overview":

    # --- Top-level summary metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")                         # number of rows
    c2.metric("Columns", f"{df.shape[1]:,}")                  # number of columns
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")  # total NA cells

    st.divider()

    # --- Preview + schema (side-by-side layout) ---
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Data Preview")
        st.dataframe(df.head(50), use_container_width=True)   # show first 50 rows

    with right:
        st.subheader("Columns & Types")

        # Build a "schema" table for user-friendly inspection
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing": df.isna().sum().values,
            "missing_%": (df.isna().mean().values * 100).round(2),
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True)

    st.divider()

    # --- Column explorer: user selects a column, app shows quick info ---
    st.subheader("Quick Column Lookups")
    col = st.selectbox("Pick a column", df.columns)

    c1, c2 = st.columns(2)

    with c1:
        # If column is NOT datetime, show top values (useful for categorical-like data)
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            st.write("Top values (categorical-like):")

            # Value counts with NA treated as "Missing"
            vc = (
                df[col]
                .fillna("Missing")
                .value_counts(dropna=False)
                .head(10)
                .rename_axis("value")
                .reset_index(name="count")
            )
            st.dataframe(vc, use_container_width=True)

        # If datetime column, skip top values (often messy/unhelpful)
        else:
            st.info("Top values are hidden for date columns.")

    with c2:
        st.write("Basic stats:")

        # Numeric column: use pandas describe()
        if pd.api.types.is_numeric_dtype(df[col]):
            st.write(df[col].describe())

        # Datetime column: show min/max + missing + monthly distribution
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dmin = df[col].min()
            dmax = df[col].max()
            missing = int(df[col].isna().sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Min date", dmin.date().isoformat() if pd.notna(dmin) else "—")
            m2.metric("Max date", dmax.date().isoformat() if pd.notna(dmax) else "—")
            m3.metric("Missing", f"{missing:,}")

            # Group by month using Periods and plot counts
            st.write("Count over time (by month)")
            monthly = df[col].dropna().dt.to_period("M").value_counts().sort_index()
            st.bar_chart(monthly)
            st.caption("Each bar shows the number of records in that month.")

        # Other types: no special stats
        else:
            st.info("Not a numeric or date column.")


# ============================================================
# PAGE 2: CLEANING REPORT
# Purpose: data quality audit (missingness, type issues, duplicates)
# ============================================================
elif page == "Cleaning Report":

    st.header("Cleaning Report")

    # High-level dataset quality summary (counts + memory)
    qs = quality_summary(df)
    a, b, c, d, e = st.columns(5)
    a.metric("Rows", f"{qs['rows']:,}")
    b.metric("Columns", f"{qs['cols']:,}")
    c.metric("Missing cells", f"{qs['missing_cells']:,}")
    d.metric("Duplicate rows", f"{qs['duplicate_rows']:,}")
    e.metric("Memory (MB)", f"{qs['memory_mb']:.2f}")

    st.divider()

    # Missing values by column (table)
    st.subheader("Missing Values by Column")
    mr = missing_report(df)
    st.dataframe(mr, use_container_width=True)

    st.divider()

    # Heuristic type issues report (e.g., numeric stored as strings)
    st.subheader("Potential Type Issues (heuristics)")
    ti = type_issues_report(df)
    if ti.empty:
        st.success("No obvious type issues detected.")
    else:
        st.dataframe(ti, use_container_width=True)

    st.divider()

    # Duplicate row detection + preview
    st.subheader("Duplicate Rows Preview")
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        st.success("No duplicate rows found.")
    else:
        st.warning(f"Found {dup_count} duplicate rows (showing up to 20).")
        st.dataframe(df[df.duplicated()].head(20), use_container_width=True)


# ============================================================
# PAGE 3: EDA DASHBOARD
# Purpose: summaries + correlations + outliers + plots + report export
# ============================================================
elif page == "EDA Dashboard":
    st.header("📈 EDA Dashboard")

    # --- Compute EDA artifacts once (reused across the page) ---
    ns = numeric_summary(df)          # numeric stats table
    cs = categorical_summary(df)      # categorical stats table
    corr = correlation_matrix(df)     # correlation matrix (numeric-only)
    out = iqr_outliers(df)            # outlier counts/rates per numeric column
    kf = key_findings(df)             # bullet insights generated by rules/heuristics

    # Key Findings section
    st.subheader("Key Findings")
    for b in kf:
        st.write("• " + b)

    st.divider()

    # Side-by-side numeric vs categorical summaries
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Numeric Summary")
        ns = numeric_summary(df)      # recomputed here (you could reuse the earlier one)
        if ns.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(ns, use_container_width=True)

    with right:
        st.subheader("Categorical Summary")
        cs = categorical_summary(df)  # recomputed here (you could reuse the earlier one)
        if cs.empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(cs, use_container_width=True)

    st.divider()

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = correlation_matrix(df)     # recomputed here (you could reuse the earlier one)

    if corr.empty:
        st.info("Need at least 2 numeric columns to compute correlations.")
    else:
        # Plotly heatmap figure from your viz module
        st.plotly_chart(corr_heatmap(corr), use_container_width=True)

    # Find the strongest absolute correlation pair (excluding self-correlation)
    strong = corr.abs().unstack().sort_values(ascending=False)
    strong = strong[strong < 1]       # remove perfect 1.0 values (diagonal)

    if not strong.empty:
        (a, b), val = strong.index[0], strong.iloc[0]
        st.caption(f"Strongest relationship: {a} ↔ {b} (|r| = {val:.2f})")

    # Outlier detection table
    st.subheader("Outlier Detection (IQR Rule)")
    out = iqr_outliers(df)            # recomputed here (you could reuse the earlier one)
    if out.empty:
        st.info("No numeric columns detected, so outliers can’t be computed.")
    else:
        st.dataframe(out, use_container_width=True)

    st.divider()

    # Distribution / column explorer
    st.subheader("Column Explorer")
    col2 = st.selectbox("Choose a column to visualize", df.columns, key="eda_col")
    fig = dist_plot(df, col2)         # returns a Plotly figure based on column type
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📥 Download Outputs")

    # Generate an HTML report string that bundles EDA outputs
    report_html = generate_eda_report(
        df=df,
        numeric_summary=ns,
        categorical_summary=cs,
        key_findings=key_findings(df),   # recompute findings (could reuse kf)
        outliers=out,
        corr=corr
    )

    # Download: HTML EDA report
    st.download_button(
        label="Download EDA Report (HTML)",
        data=report_html,
        file_name="insighthub_eda_report.html",
        mime="text/html"
    )

    # Download: cleaned dataset as CSV (whatever df currently is after optional cleaning)
    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 4: MODELING
# Purpose: pick a target and train a baseline model (auto task detect)
# ============================================================
elif page == "Modeling":
    st.header("🤖 Modeling (Baseline)")

    st.caption(
        "Pick a target column. InsightHub will auto-detect regression vs classification "
        "and train a simple baseline model."
    )

    # If only 1 column, modeling doesn’t make sense
    if df.shape[1] < 2:
        st.warning("Need at least 2 columns to build a model.")
        st.stop()

    # Select the target variable
    target = st.selectbox("Select target column (what you want to predict)", df.columns)

    # Show quick preview of target values
    st.write("Target preview:")
    st.dataframe(df[[target]].head(10), use_container_width=True)

    # Edge-case: if a class appears once, stratified split can't work well
    vc = df[target].dropna().value_counts()
    if not vc.empty and vc.min() == 1:
        st.warning(
            "Your target has at least one class that appears only once. "
            "Stratified splitting is not possible; the app will use a normal split. "
            "Consider choosing a different target or combining rare classes."
        )

    # Modeling controls: test split, model type, random seed
    colA, colB, colC = st.columns(3)
    with colA:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    with colB:
        model_choice = st.selectbox("Model", ["auto", "linear", "rf"], index=0)
    with colC:
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    # Optional feature exclusion (IDs/free-text/etc.)
    st.subheader("Optional: exclude columns")
    drop_cols = st.multiselect(
        "Drop columns (IDs, free-text, etc.)",
        options=[c for c in df.columns if c != target],
        default=[]
    )

    # Train model only when user clicks button (prevents training on every rerun)
    if st.button("Train baseline model"):
        with st.spinner("Training..."):
            # train_baseline returns a result object with task, metrics, etc.
            result = train_baseline(
                df=df,
                target=target,
                test_size=float(test_size),
                random_state=int(random_state),
                model_choice=model_choice,
                drop_cols=drop_cols
            )

        st.success(f"Done — {result.task.upper()} using {result.model_name}")

        # Display metrics as Streamlit "metric cards"
        st.subheader("Metrics")
        mcols = st.columns(min(4, len(result.metrics)))
        for i, (k, v) in enumerate(result.metrics.items()):
            if isinstance(v, float):
                mcols[i % len(mcols)].metric(k, f"{v:.4f}")
            else:
                mcols[i % len(mcols)].metric(k, str(v))

        # If classification: show confusion matrix + classification report
        if result.task == "classification" and result.confusion is not None:
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(result.confusion), use_container_width=True)

            st.subheader("Classification Report")
            st.code(result.class_report)

        # Feature importance (RF) or coefficients (linear), if available
        if result.feature_importance is not None and not result.feature_importance.empty:
            st.subheader("Top Feature Importance / Coefficients")
            st.dataframe(result.feature_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")
