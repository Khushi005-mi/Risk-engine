import pandas as pd

from engine.validation.batch_validator import BatchValidator
from engine.features.feature_pipeline import FeaturePipeline
from engine.modeling.inference_model import InferenceModel


def test_full_pipeline():

    df = pd.DataFrame({
        "customer_id": [1, 2],
        "income": [50000, 80000],
        "debt": [10000, 20000]
    })

    validator = BatchValidator()
    assert validator.validate(df)["valid"]

    pipeline = FeaturePipeline()
    features = pipeline.transform(df)

    model = InferenceModel()
    preds = model.predict(features.values)

    assert len(preds) == len(df)
    