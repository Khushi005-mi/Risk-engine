import logging
import pandas as pd

from engine.validation.batch_validator import BatchValidator
from engine.features.feature_pipeline import FeaturePipeline
from engine.modeling.inference_model import InferenceModel
from engine.governance.model_registry import ModelRegistry


class InferencePipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.validator = BatchValidator()
        self.pipeline = FeaturePipeline()
        self.model = InferenceModel()
        self.registry = ModelRegistry()

    def run(self, df: pd.DataFrame, model_name: str):

        self.logger.info("Starting inference pipeline")

        # 1. Validate
        validation = self.validator.validate(df)

        if not validation["valid"]:
            self.logger.error("Validation failed")
            raise ValueError(validation["errors"])

        # 2. Features
        features = self.pipeline.transform(df)

        # 3. Load production model
        model_info = self.registry.get_production_model(model_name)
        self.model.load(model_info["path"])

        # 4. Predict
        preds = self.model.predict_proba(features.values)

        df["prediction"] = preds

        self.logger.info("Inference completed")

        return df
    