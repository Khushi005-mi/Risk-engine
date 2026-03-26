import logging
import numpy as np

from engine.features.feature_pipeline import FeaturePipeline
from Prj_2_risk_engine.engine.modeling.gbdt_model import GBDTModel
from engine.governance.model_registry import ModelRegistry
from engine.governance.approval_workflow import ApprovalWorkflow
class TrainingPipeline:
   def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.pipeline = FeaturePipeline()
        self.model = GBDTModel()
        self.registry = ModelRegistry()
        self.approval = ApprovalWorkflow()
        class TrainingPipeline:
        
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
   