import logging
import pandas as pd
import numpy as np

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


class SurvivalModel:
    """
    Survival model for predicting time-to-default using Cox Proportional Hazards.
    """

    def __init__(self):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.model = CoxPHFitter()
        self.fitted = False


    def prepare_dataset(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        feature_cols: list
    ) -> pd.DataFrame:
        """
        Prepare dataset for survival modeling.
        """

        self.logger.info("Preparing dataset for survival model")

        data = df[feature_cols + [duration_col, event_col]].copy()

        data = data.dropna()

        return data


    def train(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str
    ):
        """
        Train Cox Proportional Hazards model.
        """

        self.logger.info("Training survival model")

        self.model.fit(
            df,
            duration_col=duration_col,
            event_col=event_col
        )

        self.fitted = True

        self.logger.info("Model training completed")


    def predict_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict relative risk score for each observation.
        """

        if not self.fitted:
            raise ValueError("Model must be trained before prediction")

        self.logger.info("Predicting risk scores")

        risk_scores = self.model.predict_partial_hazard(df)

        return risk_scores


    def predict_survival_function(self, df: pd.DataFrame):
        """
        Predict survival probability curves.
        """

        if not self.fitted:
            raise ValueError("Model must be trained before prediction")

        self.logger.info("Predicting survival curves")

        survival_curves = self.model.predict_survival_function(df)

        return survival_curves


    def evaluate(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str
    ):
        """
        Evaluate model using concordance index.
        """

        self.logger.info("Evaluating survival model")

        risk_scores = self.model.predict_partial_hazard(df)

        c_index = concordance_index(
            df[duration_col],
            -risk_scores,
            df[event_col]
        )

        self.logger.info(f"C-index: {c_index}")

        return c_index


    def save_model(self, path: str):

        self.logger.info(f"Saving survival model to {path}")

        self.model.save(path)


    def load_model(self, path: str):

        self.logger.info(f"Loading survival model from {path}")

        self.model = CoxPHFitter().load(path)

        self.fitted = True

        