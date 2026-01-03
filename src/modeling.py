from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

@dataclass
class ModelResult:
    task: str  # "regression" or "classification"
    model_name: str
    metrics: Dict[str, Any]
    pipeline: Any
    feature_importance: Optional[pd.DataFrame] = None
    confusion: Optional[np.ndarray] = None
    class_report: Optional[str] = None


def infer_task(y: pd.Series, max_classes: int = 20) -> str:
    """Heuristic: numeric with many unique -> regression else classification."""
    if pd.api.types.is_numeric_dtype(y):
        uniq = y.dropna().nunique()
        # numeric but small unique counts often means labels (0/1/2)
        if uniq <= max_classes:
            return "classification"
        return "regression"
    return "classification"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    # treat datetime as categorical-ish by string (simple + robust)
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    # Convert datetime columns to string upstream (we’ll do it in make_xy)
    # so here we only expect numeric + categorical.
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols + dt_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre


def make_xy(df: pd.DataFrame, target: str, drop_cols: Optional[List[str]] = None):
    drop_cols = drop_cols or []
    df = df.copy()

    y = df[target]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")

    # Convert datetime columns to string so OneHotEncoder can handle them
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
    X, y = make_xy(df, target, drop_cols=drop_cols)

    # Drop rows with missing target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    task = infer_task(y)

    stratify_arg = None
    if task == "classification":
        vc = pd.Series(y).value_counts(dropna=True)
        # stratify only if every class has at least 2 samples
        if not vc.empty and vc.min() >= 2:
            stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )


    pre = _build_preprocessor(X_train)

    if task == "regression":
        if model_choice == "rf":
            model = RandomForestRegressor(n_estimators=250, random_state=random_state)
            model_name = "RandomForestRegressor"
        else:
            model = LinearRegression()
            model_name = "LinearRegression"

        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

        fi = extract_feature_importance(pipe, top_k=20)

        return ModelResult(
            task="regression",
            model_name=model_name,
            metrics=metrics,
            pipeline=pipe,
            feature_importance=fi
        )

    # classification
    # If too many classes, RF generally behaves better than logistic
    n_classes = int(pd.Series(y_train).nunique())
    use_rf = (model_choice == "rf") or (model_choice == "auto" and n_classes > 2)

    if use_rf:
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
        model_name = "RandomForestClassifier"
    else:
        model = LogisticRegression(max_iter=2000)
        model_name = "LogisticRegression"

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

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
    try:
        pre: ColumnTransformer = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]
    except Exception:
        return None

    # Get output feature names from the preprocessor
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        s = pd.Series(vals, index=feature_names).sort_values(ascending=False).head(top_k)
        return s.reset_index().rename(columns={"index": "feature", 0: "importance"})
    if hasattr(model, "coef_"):
        coef = model.coef_
        # multiclass: average abs across classes
        if coef.ndim == 2:
            vals = np.mean(np.abs(coef), axis=0)
        else:
            vals = np.abs(coef)
        s = pd.Series(vals, index=feature_names).sort_values(ascending=False).head(top_k)
        return s.reset_index().rename(columns={"index": "feature", 0: "importance"})
    return None