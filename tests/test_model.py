"""Unit tests for data validation and preprocessing/model behaviour."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from titanic_tree.model import FEATURES, TARGET, build_pipeline, load_dataset, original_feature_importance


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Survived": [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
            "Pclass": [3, 1, 2, 3, 3, 1, 2, 1, 3, 2],
            "Age": [22.0, 38.0, np.nan, 35.0, 30.0, 2.0, 54.0, 27.0, 40.0, 14.0],
            "Fare": [7.25, 71.3, 13.0, 8.05, 7.9, 21.0, 26.0, 50.0, 8.5, 30.0],
            "Sex": ["male", "female", "female", "male", "male", "female", "male", "female", "male", "female"],
        }
    )


def test_pipeline_handles_missing_age_and_predicts():
    df = sample_frame()
    model = build_pipeline(max_depth=3, min_samples_leaf=1)
    model.fit(df[FEATURES], df[TARGET])
    predictions = model.predict(df[FEATURES])
    assert len(predictions) == len(df)
    assert set(predictions).issubset({0, 1})


def test_feature_importance_is_aggregated_to_four_inputs():
    df = sample_frame()
    model = build_pipeline(max_depth=3, min_samples_leaf=1).fit(df[FEATURES], df[TARGET])
    importance = original_feature_importance(model)
    assert set(importance["feature"]) == set(FEATURES)
    assert np.isclose(importance["importance"].sum(), 1.0)


def test_load_dataset_rejects_missing_required_columns(tmp_path):
    file = tmp_path / "bad.csv"
    pd.DataFrame({"Survived": [0], "Age": [10]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(file)


def test_load_dataset_accepts_standard_titanic_columns(tmp_path):
    file = tmp_path / "train.csv"
    sample_frame().to_csv(file, index=False)
    loaded = load_dataset(file)
    assert len(loaded) == len(sample_frame())
    assert set(FEATURES + [TARGET]).issubset(loaded.columns)


def test_end_to_end_run_writes_metrics_and_figures(tmp_path):
    from titanic_tree.model import train_and_evaluate

    data = pd.concat([sample_frame()] * 5, ignore_index=True)
    csv_path = tmp_path / "train.csv"
    data.to_csv(csv_path, index=False)
    output_dir = tmp_path / "reports"
    model_path = tmp_path / "models" / "pipeline.joblib"

    result = train_and_evaluate(
        data_path=csv_path,
        output_dir=output_dir,
        model_path=model_path,
        max_depth=3,
        min_samples_leaf=2,
    )

    assert result["metrics"]["dataset_rows"] == len(data)
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "figures" / "feature_importance.png").exists()
    assert model_path.exists()
