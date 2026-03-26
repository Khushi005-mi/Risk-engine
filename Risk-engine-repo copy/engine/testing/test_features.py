import pandas as pd

from engine.features.feature_pipeline import FeaturePipeline


def test_feature_generation():

    df = pd.DataFrame({
        "income": [50000, 100000],
        "debt": [10000, 20000]
    })

    pipeline = FeaturePipeline()

    result = pipeline.transform(df)

    assert "debt_to_income_ratio" in result.columns


def test_no_nan_in_features():

    df = pd.DataFrame({
        "income": [50000, None],
        "debt": [10000, 20000]
    })

    pipeline = FeaturePipeline()

    result = pipeline.transform(df)

    assert result.isnull().sum().sum() == 0