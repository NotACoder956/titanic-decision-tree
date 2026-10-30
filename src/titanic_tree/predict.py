"""Predict survival for one passenger using a previously trained pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one passenger's survival class.")
    parser.add_argument("--model", default="models/decision_tree_pipeline.joblib")
    parser.add_argument("--sex", choices=["female", "male"], required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--pclass", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--fare", type=float, required=True)
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model not found at {model_path}. Run the training command first.")
    if args.age < 0 or args.fare < 0:
        parser.error("Age and fare must be non-negative.")
    model = joblib.load(model_path)
    passenger = pd.DataFrame([{"Pclass": args.pclass, "Age": args.age, "Fare": args.fare, "Sex": args.sex}])
    prediction = int(model.predict(passenger)[0])
    label = "Survived" if prediction == 1 else "Did not survive"
    print(f"Prediction: {label} (Survived={prediction})")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(passenger)[0]
        print(f"Estimated probability of survival: {probabilities[1]:.1%}")
        print("Note: this is a model estimate, not a guarantee or a causal conclusion.")


if __name__ == "__main__":
    main()
