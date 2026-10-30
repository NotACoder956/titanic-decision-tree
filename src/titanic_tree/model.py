"""Reusable data preparation, training, and evaluation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

TARGET = "Survived"
NUMERIC_FEATURES = ["Pclass", "Age", "Fare"]
CATEGORICAL_FEATURES = ["Sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the labelled Kaggle training file and validate required columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download Kaggle's train.csv and place it at data/train.csv, "
            "or pass its location with --data. See data/README.md."
        )
    df = pd.read_csv(path)
    # Kaggle uses title case; accepting lower-case variants makes the loader a little more robust.
    rename = {col: col[:1].upper() + col[1:] for col in df.columns if col.lower() in {"survived", "pclass", "sex", "age", "fare"}}
    df = df.rename(columns=rename)
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[TARGET].isna().any():
        raise ValueError("Target column 'Survived' contains missing values; remove those rows before training.")
    values = set(pd.to_numeric(df[TARGET], errors="coerce").dropna().unique())
    if not values.issubset({0, 1}):
        raise ValueError("Target column 'Survived' must contain only 0 and 1.")
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_pipeline(max_depth: int = 3, min_samples_leaf: int = 8) -> Pipeline:
    """Build preprocessing + tree as one leakage-safe sklearn pipeline."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def original_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Aggregate one-hot encoded category importances back to the original input features."""
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    aggregated = {feature: 0.0 for feature in FEATURES}
    for transformed_name, importance in zip(transformed_names, importances):
        # ColumnTransformer names are e.g. num__Age and cat__Sex_female.
        clean_name = transformed_name.split("__", 1)[-1]
        if clean_name.startswith("Sex_") or clean_name == "Sex":
            original_name = "Sex"
        else:
            original_name = clean_name
        if original_name in aggregated:
            aggregated[original_name] += float(importance)
    result = pd.DataFrame(
        [{"feature": name, "importance": value} for name, value in aggregated.items()]
    )
    return result.sort_values("importance", ascending=False, ignore_index=True)


def _plot_data_overview(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    survival_counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    axes[0].bar(["Did not survive", "Survived"], survival_counts.values)
    axes[0].set_title("Survival labels in the dataset")
    axes[0].set_ylabel("Number of passengers")
    axes[0].tick_params(axis="x", rotation=10)
    for i, value in enumerate(survival_counts.values):
        axes[0].text(i, value + max(survival_counts.values) * 0.015, str(value), ha="center")

    missing = df[FEATURES].isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        axes[1].text(0.5, 0.5, "No missing values in selected features", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        axes[1].bar(missing.index.astype(str), missing.values)
        axes[1].set_title("Missing values before preprocessing")
        axes[1].set_ylabel("Missing entries")
        for i, value in enumerate(missing.values):
            axes[1].text(i, value + max(missing.values) * 0.015, str(value), ha="center")
    fig.suptitle("Dataset overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = importance_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title("Which input features influenced the tree?", fontweight="bold")
    ax.set_xlabel("Aggregated decision-tree feature importance")
    ax.set_ylabel("Input feature")
    ax.set_xlim(0, max(0.05, float(plot_df["importance"].max()) * 1.2))
    for bar, value in zip(bars, plot_df["importance"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Predicted: No", "Predicted: Yes"])
    ax.set_yticks([0, 1], labels=["Actual: No", "Actual: Yes"])
    ax.set_title("Validation confusion matrix", fontweight="bold")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > threshold else "black", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_tree(pipeline: Pipeline, output_path: Path) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = [name.split("__", 1)[-1].replace("Sex_", "Sex = ") for name in preprocessor.get_feature_names_out()]
    fig, ax = plt.subplots(figsize=(17, 9))
    plot_tree(
        classifier,
        feature_names=names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=2,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Interpretable Decision Tree (limited depth)", fontsize=15, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(
    data_path: str | Path,
    output_dir: str | Path = "reports",
    model_path: str | Path = "models/decision_tree_pipeline.joblib",
    max_depth: int = 3,
    min_samples_leaf: int = 8,
    test_size: float = 0.20,
) -> dict[str, Any]:
    """Train and evaluate the model; write reproducible metrics and charts."""
    import joblib

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.10 and 0.40 for this project.")
    df = load_dataset(data_path)
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = build_pipeline(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)

    majority_label = int(y_train.value_counts().idxmax())
    baseline_accuracy = float((y_valid == majority_label).mean())
    metrics = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "validation_split": float(test_size),
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
        "precision_survived": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall_survived": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1_survived": float(f1_score(y_valid, predictions, zero_division=0)),
        "majority_class_baseline_accuracy": baseline_accuracy,
        "baseline_accuracy_gap": float(accuracy_score(y_valid, predictions) - baseline_accuracy),
    }
    importance_df = original_feature_importance(pipeline)
    missing = {str(key): int(value) for key, value in df[FEATURES].isna().sum().items()}
    class_distribution = {str(int(key)): int(value) for key, value in y.value_counts().sort_index().items()}
    report_text = classification_report(
        y_valid, predictions, labels=[0, 1], target_names=["Did not survive", "Survived"], zero_division=0
    )
    result: dict[str, Any] = {
        "project": "Titanic Survival Prediction with a Decision Tree",
        "metrics": metrics,
        "model_config": {"algorithm": "DecisionTreeClassifier", "criterion": "gini", "max_depth": max_depth, "min_samples_leaf": min_samples_leaf},
        "features": FEATURES,
        "target": TARGET,
        "class_distribution": class_distribution,
        "missing_values_before_preprocessing": missing,
        "feature_importance": [
            {"feature": row.feature, "importance": float(row.importance)}
            for row in importance_df.itertuples(index=False)
        ],
        "confusion_matrix_labels": ["Did not survive", "Survived"],
        "confusion_matrix": confusion_matrix(y_valid, predictions, labels=[0, 1]).tolist(),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "classification_report.txt").write_text(report_text, encoding="utf-8")
    importance_df.to_csv(output / "feature_importance.csv", index=False)
    pd.DataFrame({"actual_survived": y_valid.to_numpy(), "predicted_survived": predictions}).to_csv(
        output / "validation_predictions.csv", index=False
    )
    _plot_data_overview(df, figures / "dataset_overview.png")
    _plot_feature_importance(importance_df, figures / "feature_importance.png")
    _plot_confusion_matrix(y_valid, predictions, figures / "confusion_matrix.png")
    _plot_tree(pipeline, figures / "decision_tree.png")
    joblib.dump(pipeline, model_file)
    _write_summary(result, output / "SUMMARY.md")
    return result


def _write_summary(result: dict[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    best = result["feature_importance"][0]
    lines = [
        "# Experiment summary", "",
        "This file is generated by `python -m titanic_tree.train`. It records the results for the dataset and configuration used in that run.", "",
        "## Evaluation", "",
        f"- Dataset rows: **{metrics['dataset_rows']}**",
        f"- Training / validation rows: **{metrics['training_rows']} / {metrics['validation_rows']}**",
        f"- Validation accuracy: **{metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})**",
        f"- Balanced accuracy: **{metrics['balanced_accuracy']:.3f}**",
        f"- Precision for survival class: **{metrics['precision_survived']:.3f}**",
        f"- Recall for survival class: **{metrics['recall_survived']:.3f}**",
        f"- F1-score for survival class: **{metrics['f1_survived']:.3f}**",
        f"- Majority-class baseline accuracy: **{metrics['majority_class_baseline_accuracy']:.3f} ({metrics['majority_class_baseline_accuracy']:.1%})**",
        "", "## Feature importance", "",
    ]
    lines.extend(f"- **{row['feature']}**: {row['importance']:.3f}" for row in result["feature_importance"])
    lines += ["", f"The top-ranked feature in this run is **{best['feature']}**. Feature importance reflects the tree's use of each input; it does not establish causation.", "", "## Missing values", ""]
    lines.extend(f"- {feature}: {count}" for feature, count in result["missing_values_before_preprocessing"].items())
    lines += ["", "## Figures", "", "- `figures/dataset_overview.png`", "- `figures/feature_importance.png`", "- `figures/confusion_matrix.png`", "- `figures/decision_tree.png`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
