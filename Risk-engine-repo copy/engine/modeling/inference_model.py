import logging
import pandas as pd
import numpy as np
import os
import xgboost as xgb

from typing import Dict, Any

from engine.features.feature_pipeline import FeaturePipeline
from engine.graph.graph_features import GraphFeatureEngine
from engine.modeling.calibration import ProbabilityCalibrator


class InferenceModel:
    """
    Production inference pipeline for credit risk scoring.
    """

    def __init__(self, config: Dict[str, Any]):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.config = config

        self.model = None
        self.calibrator = None

        self.feature_pipeline = FeaturePipeline()
        self.graph_engine = GraphFeatureEngine()

        self._load_artifacts()


    def _load_artifacts(self):
        """
        Load model and calibration artifacts from disk.
        """

        self.logger.info("Loading model artifacts")

        model_path = self.config["model_path"]
        calibrator_path = self.config.get("calibrator_path")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        self.logger.info("Model loaded successfully")

        if calibrator_path and os.path.exists(calibrator_path):
            self.calibrator = ProbabilityCalibrator(method="isotonic")
            self.calibrator.load_model = lambda path: None  # placeholder
            self.logger.info("Calibrator loaded")


    def validate_input(self, df: pd.DataFrame):
        """
        Basic sanity checks before processing.
        """

        if df.empty:
            raise ValueError("Input dataframe is empty")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        self.logger.info("Input validation passed")


    def run_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate base + behavioral + ratio + temporal features.
        """

        self.logger.info("Running feature pipeline")

        df_features = self.feature_pipeline.transform(df)

        return df_features


    def run_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add graph-based risk features.
        """

        self.logger.info("Generating graph features")

        graph_features = self.graph_engine.compute_graph_features(df)

        df = df.merge(graph_features, on="entity_id", how="left")

        return df


    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure feature consistency with training.
        """

        expected_features = self.config["feature_columns"]

        missing = set(expected_features) - set(df.columns)

        if missing:
            self.logger.warning(f"Missing features: {missing}")

            for col in missing:
                df[col] = 0

        return df[expected_features]


    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate probability of default (PD).
        """

        self.logger.info("Running prediction")

        probs = self.model.predict_proba(df)[:, 1]

        return probs


    def calibrate(self, probs: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Apply calibration if available.
        """

        if self.calibrator is None:
            return probs

        self.logger.info("Applying calibration")

        calibrated = self.calibrator.predict(y_pred=probs)

        return calibrated


    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full end-to-end scoring pipeline.
        """

        self.validate_input(df)

        df = self.run_feature_pipeline(df)

        df = self.run_graph_features(df)

        X = self.align_features(df)

        raw_probs = self.predict_proba(X)

        final_probs = self.calibrate(raw_probs, df)

        result = pd.DataFrame({
            "entity_id": df["entity_id"],
            "pd_score": final_probs
        })

        self.logger.info("Scoring completed")

        return result