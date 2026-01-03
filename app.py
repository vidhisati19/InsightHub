import streamlit as st
import pandas as pd
from src.basic_clean import basic_prepare
from src.data_load import load_csv
from src.quality_report import quality_summary, missing_report, type_issues_report
from src.cleaning_actions import apply_cleaning
from src.eda import numeric_summary, categorical_summary, correlation_matrix, iqr_outliers, key_findings
from src.viz import dist_plot, corr_heatmap
from src.modeling import train_baseline
from src.reporting import generate_eda_report


st.set_page_config(
    page_title="InsightHub",
    page_icon="📊",
    layout="wide",
)

st.title("📊 InsightHub")
st.caption("Upload a CSV to instantly preview and understand your dataset.")

# --- Sidebar navigation
page = st.sidebar.radio("Navigate", ["Overview", "Cleaning Report", "EDA Dashboard", "Modeling"], index=0)
st.sidebar.divider()

# --- Upload (shared by all pages)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# --- Load + basic prep (shared by all pages)
df = load_csv(uploaded_file)
df = basic_prepare(df)

# --- Sidebar cleaning options (shared by all pages)
st.sidebar.subheader("Cleaning Options")
drop_dupes = st.sidebar.checkbox("Drop duplicate rows", value=True)

fill_num = st.sidebar.selectbox(
    "Fill missing numeric values",
    options=["None", "Median", "Mean"],
    index=0
)

fill_cat = st.sidebar.selectbox(
    "Fill missing categorical values",
    options=["None", "Mode", "Unknown"],
    index=0
)

apply_opts = st.sidebar.checkbox("Apply cleaning to dataset", value=False)

if apply_opts:
    df = apply_cleaning(
        df,
        drop_duplicates=drop_dupes,
        fill_numeric=None if fill_num == "None" else fill_num.lower(),
        fill_categorical=None if fill_cat == "None" else fill_cat.lower(),
    )

# =========================
# PAGE 1: OVERVIEW
# =========================
if page == "Overview":

    # --- Top-level summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    st.divider()

    # --- Preview + schema
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Data Preview")
        st.dataframe(df.head(50), use_container_width=True)

    with right:
        st.subheader("Columns & Types")
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing": df.isna().sum().values,
            "missing_%": (df.isna().mean().values * 100).round(2),
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True)

    st.divider()

    # --- Quick selectors
    st.subheader("Quick Column Lookups")
    col = st.selectbox("Pick a column", df.columns)

    c1, c2 = st.columns(2)

    with c1:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            st.write("Top values (categorical-like):")
            vc = (
                df[col]
                .fillna("Missing")
                .value_counts(dropna=False)
                .head(10)
                .rename_axis("value")
                .reset_index(name="count")
            )
            st.dataframe(vc, use_container_width=True)
        else:
            st.info("Top values are hidden for date columns.")

    with c2:
        st.write("Basic stats:")
        if pd.api.types.is_numeric_dtype(df[col]):
            st.write(df[col].describe())

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dmin = df[col].min()
            dmax = df[col].max()
            missing = int(df[col].isna().sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Min date", dmin.date().isoformat() if pd.notna(dmin) else "—")
            m2.metric("Max date", dmax.date().isoformat() if pd.notna(dmax) else "—")
            m3.metric("Missing", f"{missing:,}")

            st.write("Count over time (by month)")
            monthly = df[col].dropna().dt.to_period("M").value_counts().sort_index()
            st.bar_chart(monthly)
            st.caption("Each bar shows the number of records in that month.")

        else:
            st.info("Not a numeric or date column.")

# =========================
# PAGE 2: CLEANING REPORT
# =========================
elif page == "Cleaning Report":

    st.header("🧼 Cleaning Report")

    qs = quality_summary(df)
    a, b, c, d, e = st.columns(5)
    a.metric("Rows", f"{qs['rows']:,}")
    b.metric("Columns", f"{qs['cols']:,}")
    c.metric("Missing cells", f"{qs['missing_cells']:,}")
    d.metric("Duplicate rows", f"{qs['duplicate_rows']:,}")
    e.metric("Memory (MB)", f"{qs['memory_mb']:.2f}")

    st.divider()

    st.subheader("Missing Values by Column")
    mr = missing_report(df)
    st.dataframe(mr, use_container_width=True)

    st.divider()

    st.subheader("Potential Type Issues (heuristics)")
    ti = type_issues_report(df)
    if ti.empty:
        st.success("No obvious type issues detected.")
    else:
        st.dataframe(ti, use_container_width=True)

    st.divider()

    st.subheader("Duplicate Rows Preview")
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        st.success("No duplicate rows found.")
    else:
        st.warning(f"Found {dup_count} duplicate rows (showing up to 20).")
        st.dataframe(df[df.duplicated()].head(20), use_container_width=True)

elif page == "EDA Dashboard":
    st.header("📈 EDA Dashboard")
    # --- Compute EDA artifacts once (reuse everywhere)
    ns = numeric_summary(df)
    cs = categorical_summary(df)
    corr = correlation_matrix(df)
    out = iqr_outliers(df)
    kf = key_findings(df)

    # Key Findings
    st.subheader("Key Findings")
    for b in kf:
        st.write("• " + b)

    st.divider()

    # Numeric + categorical summaries
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Numeric Summary")
        ns = numeric_summary(df)
        if ns.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(ns, use_container_width=True)

    with right:
        st.subheader("Categorical Summary")
        cs = categorical_summary(df)
        if cs.empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(cs, use_container_width=True)

    st.divider()

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = correlation_matrix(df)

    if corr.empty:
        st.info("Need at least 2 numeric columns to compute correlations.")
    else:
        st.plotly_chart(corr_heatmap(corr), use_container_width=True)

    # ---- strongest correlation (UI logic belongs HERE)
    strong = corr.abs().unstack().sort_values(ascending=False)
    strong = strong[strong < 1]

    if not strong.empty:
        (a, b), val = strong.index[0], strong.iloc[0]
        st.caption(f"Strongest relationship: {a} ↔ {b} (|r| = {val:.2f})")

    # Outliers
    st.subheader("Outlier Detection (IQR Rule)")
    out = iqr_outliers(df)
    if out.empty:
        st.info("No numeric columns detected, so outliers can’t be computed.")
    else:
        st.dataframe(out, use_container_width=True)

    st.divider()

    # Distribution / column explorer
    st.subheader("Column Explorer")
    col2 = st.selectbox("Choose a column to visualize", df.columns, key="eda_col")
    fig = dist_plot(df, col2)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📥 Download Outputs")

    # Generate report HTML
    report_html = generate_eda_report(
        df=df,
        numeric_summary=ns,
        categorical_summary=cs,
        key_findings=key_findings(df),
        outliers=out,
        corr=corr
    )

    st.download_button(
        label="Download EDA Report (HTML)",
        data=report_html,
        file_name="insighthub_eda_report.html",
        mime="text/html"
    )

    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

elif page == "Modeling":
    st.header("🤖 Modeling (Baseline)")

    st.caption("Pick a target column. InsightHub will auto-detect regression vs classification and train a simple baseline model.")

    if df.shape[1] < 2:
        st.warning("Need at least 2 columns to build a model.")
        st.stop()

    target = st.selectbox("Select target column (what you want to predict)", df.columns)
    st.write("Target preview:")
    st.dataframe(df[[target]].head(10), use_container_width=True)

    # Target sanity check (classification edge case)
    vc = df[target].dropna().value_counts()
    if not vc.empty and vc.min() == 1:
        st.warning(
            "Your target has at least one class that appears only once. "
            "Stratified splitting is not possible; the app will use a normal split. "
            "Consider choosing a different target or combining rare classes."
        )

    colA, colB, colC = st.columns(3)
    with colA:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    with colB:
        model_choice = st.selectbox("Model", ["auto", "linear", "rf"], index=0)
    with colC:
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    # allow dropping obvious ID columns
    st.subheader("Optional: exclude columns")
    drop_cols = st.multiselect(
        "Drop columns (IDs, free-text, etc.)",
        options=[c for c in df.columns if c != target],
        default=[]
    )

    if st.button("Train baseline model"):
        with st.spinner("Training..."):
            result = train_baseline(
                df=df,
                target=target,
                test_size=float(test_size),
                random_state=int(random_state),
                model_choice=model_choice,
                drop_cols=drop_cols
            )

        st.success(f"Done — {result.task.upper()} using {result.model_name}")

        st.subheader("Metrics")
        mcols = st.columns(min(4, len(result.metrics)))
        for i, (k, v) in enumerate(result.metrics.items()):
            if isinstance(v, float):
                mcols[i % len(mcols)].metric(k, f"{v:.4f}")
            else:
                mcols[i % len(mcols)].metric(k, str(v))

        if result.task == "classification" and result.confusion is not None:
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(result.confusion), use_container_width=True)

            st.subheader("Classification Report")
            st.code(result.class_report)

        if result.feature_importance is not None and not result.feature_importance.empty:
            st.subheader("Top Feature Importance / Coefficients")
            st.dataframe(result.feature_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")