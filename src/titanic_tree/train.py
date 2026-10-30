"""Command-line entry point for the Titanic Decision Tree project."""
from __future__ import annotations

import argparse
from pathlib import Path

from titanic_tree.model import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a Titanic survival Decision Tree.")
    parser.add_argument("--data", default="data/train.csv", help="Path to Kaggle's labelled train.csv")
    parser.add_argument("--output-dir", default="reports", help="Directory for metrics and figures")
    parser.add_argument("--model-path", default="models/decision_tree_pipeline.joblib", help="Where to save the fitted pipeline")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum tree depth (default: 3)")
    parser.add_argument("--min-samples-leaf", type=int, default=8, help="Minimum samples per leaf (default: 8)")
    args = parser.parse_args()
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if args.min_samples_leaf < 1:
        parser.error("--min-samples-leaf must be at least 1")

    result = train_and_evaluate(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        model_path=Path(args.model_path),
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    metrics = result["metrics"]
    print("\nTITANIC DECISION TREE — RESULTS")
    print("=" * 38)
    print(f"Rows loaded:              {metrics['dataset_rows']}")
    print(f"Training rows:            {metrics['training_rows']}")
    print(f"Validation rows:          {metrics['validation_rows']}")
    print(f"Validation accuracy:      {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print(f"Balanced accuracy:        {metrics['balanced_accuracy']:.3f}")
    print(f"Precision (survived):     {metrics['precision_survived']:.3f}")
    print(f"Recall (survived):        {metrics['recall_survived']:.3f}")
    print(f"F1-score (survived):      {metrics['f1_survived']:.3f}")
    print(f"Majority baseline:        {metrics['majority_class_baseline_accuracy']:.3f} ({metrics['majority_class_baseline_accuracy']:.1%})")
    print("\nFeature importance:")
    for item in result["feature_importance"]:
        print(f"  {item['feature']:<8} {item['importance']:.3f}")
    print("\nSaved outputs to reports/ and the fitted model to models/.")


if __name__ == "__main__":
    main()
