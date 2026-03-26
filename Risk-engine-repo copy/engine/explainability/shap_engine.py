# Final Prediction = Base Value + Sum of Feature 
import logging
import pandas as pd
import numpy as np
import shap_engine
import xgboost as xgb

from typing import Dict, Any, List


class SHAPExplainer:
    """
    SHAP-based explainability engine for model predictions.
    """

    def __init__(self, model: xgb.XGBClassifier):

        self.logger = logging.getLogger(__name__) #getLogger(__name__) → creates a logger for this file so you know where logs come from

#basicConfig(level=INFO) → tells Python to actually show logs of level INFO and above
        logging.basicConfig(level=logging.INFO)

        self.model = model

        self.explainer = shap_engine.TreeExplainer(self.model)

        self.logger.info("SHAP Explainer initialized")


    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for dataset.
        """

        self.logger.info("Computing SHAP values")

        shap_values = self.explainer.shap_values(X)

        return shap_values


    def explain_prediction(
        self,
        X: pd.DataFrame,
        index: int
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        """

        shap_values = self.compute_shap_values(X)

        row_values = shap_values[index]
        row_data = X.iloc[index]

        explanation = []

        for feature, value, contribution in zip(
            X.columns,
            row_data.values,
            row_values
        ):
            explanation.append({
                "feature": feature,
                "value": float(value),
                "contribution": float(contribution)
            })

        explanation_sorted = sorted(
            explanation,
            key=lambda x: abs(x["contribution"]),
            reverse=True
        )

        base_value = self.explainer.expected_value

        return {
            "base_value": float(base_value),
            "prediction_contributions": explanation_sorted
        }


    def global_feature_importance(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute global feature importance using mean absolute SHAP values.
        """

        shap_values = self.compute_shap_values(X)

        importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        })

        df = df.sort_values("importance", ascending=False)

        return df


    def get_top_features(
        self,
        X: pd.DataFrame,
        index: int,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top N contributing features for a prediction.
        """

        explanation = self.explain_prediction(X, index)

        return explanation["prediction_contributions"][:top_n]