from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Scikit-learn building blocks
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Metrics
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

# ----------------------------
# Result container (clean API)
# ----------------------------
@dataclass
class ModelResult:
    task: str  # "regression" or "classification"
    model_name: str
    metrics: Dict[str, Any]
    pipeline: Any  # fitted sklearn Pipeline
    feature_importance: Optional[pd.DataFrame] = None
    confusion: Optional[np.ndarray] = None
    class_report: Optional[str] = None


def infer_task(y: pd.Series, max_classes: int = 20) -> str:
    """Heuristic: numeric with many unique -> regression else classification."""
    # If numeric dtype, decide based on how many unique values it has
    if pd.api.types.is_numeric_dtype(y):
        uniq = y.dropna().nunique()

        # Numeric but low unique count often means labels like 0/1/2 -> classification
        if uniq <= max_classes:
            return "classification"

        # Numeric with many unique values behaves like a continuous target -> regression
        return "regression"

    # Non-numeric targets default to classification
    return "classification"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify column groups for different preprocessing
    num_cols = X.select_dtypes(include="number").columns.tolist()

    # datetime columns will be treated like categorical (as strings)
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    # typical categorical columns
    cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    # Numeric preprocessing:
    # - Fill missing with median
    # - Scale numeric features (with_mean=False keeps sparse compatibility)
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    # Categorical preprocessing:
    # - Fill missing with most frequent
    # - One-hot encode categories (ignore unseen categories at test time)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    # Combine numeric + categorical pipelines into one transformer
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols + dt_cols),
        ],
        remainder="drop",       # drop any columns not listed above
        sparse_threshold=0.3    # keep sparse representation when appropriate
    )
    return pre


def make_xy(df: pd.DataFrame, target: str, drop_cols: Optional[List[str]] = None):
    # Drop columns optionally (IDs, free-text, leakage columns)
    drop_cols = drop_cols or []
    df = df.copy()

    # Separate features and target
    y = df[target]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")

    # Convert datetime columns to string so OneHotEncoder can treat them as categories
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = X[c].dt.date.astype(str)

    return X, y


def train_baseline(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    model_choice: str = "auto",  # "auto" | "linear" | "rf"
    drop_cols: Optional[List[str]] = None,
) -> ModelResult:
    # Build X (features) and y (target)
    X, y = make_xy(df, target, drop_cols=drop_cols)

    # Remove rows where the target is missing
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Decide regression vs classification
    task = infer_task(y)

    # Stratify for classification (only if each class has at least 2 samples)
    stratify_arg = None
    if task == "classification":
        vc = pd.Series(y).value_counts(dropna=True)
        if not vc.empty and vc.min() >= 2:
            stratify_arg = y

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    # Build preprocessing transformer based on training features
    pre = _build_preprocessor(X_train)

    # --------------------
    # Regression branch
    # --------------------
    if task == "regression":
        # Choose model type
        if model_choice == "rf":
            model = RandomForestRegressor(n_estimators=250, random_state=random_state)
            model_name = "RandomForestRegressor"
        else:
            model = LinearRegression()
            model_name = "LinearRegression"

        # Full pipeline: preprocess -> model
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        # Predict + compute metrics
        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

        # Try to extract feature importance / coefficients
        fi = extract_feature_importance(pipe, top_k=20)

        return ModelResult(
            task="regression",
            model_name=model_name,
            metrics=metrics,
            pipeline=pipe,
            feature_importance=fi
        )

    # ------------------------
    # Classification branch
    # ------------------------
    n_classes = int(pd.Series(y_train).nunique())

    # Auto rule:
    # - for many classes, random forest often works better than logistic regression
    use_rf = (model_choice == "rf") or (model_choice == "auto" and n_classes > 2)

    if use_rf:
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
        model_name = "RandomForestClassifier"
    else:
        model = LogisticRegression(max_iter=2000)
        model_name = "LogisticRegression"

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # Predict + classification metrics
    preds = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="weighted"))
    cm = confusion_matrix(y_test, preds)

    metrics = {"Accuracy": acc, "F1_weighted": f1, "Classes": n_classes}
    report = classification_report(y_test, preds)

    fi = extract_feature_importance(pipe, top_k=20)

    return ModelResult(
        task="classification",
        model_name=model_name,
        metrics=metrics,
        pipeline=pipe,
        feature_importance=fi,
        confusion=cm,
        class_report=report
    )


def extract_feature_importance(pipe: Pipeline, top_k: int = 20) -> Optional[pd.DataFrame]:
    """
    Returns top_k feature importances:
    - For RandomForest: feature_importances_
    - For Linear/Logistic: absolute coefficient magnitude
    """
    # Pull out the fitted preprocessor + model from the pipeline
    try:
        pre: ColumnTransformer = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]
    except Exception:
        return None

    # Get expanded feature names after preprocessing
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        return None

    # RandomForest: built-in importance scores
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        s = pd.Series(vals, index=feature_names).sort_values(ascending=False).head(top_k)
        return s.reset_index().rename(columns={"index": "feature", 0: "importance"})

    # Linear/Logistic: coefficients (importance ~ magnitude)
    if hasattr(model, "coef_"):
        coef = model.coef_

        # Multiclass case: average magnitude across classes
        if coef.ndim == 2:
            vals = np.mean(np.abs(coef), axis=0)
        else:
            vals = np.abs(coef)

        s = pd.Series(vals, index=feature_names).sort_values(ascending=False).head(top_k)
        return s.reset_index().rename(columns={"index": "feature", 0: "importance"})

    return None
