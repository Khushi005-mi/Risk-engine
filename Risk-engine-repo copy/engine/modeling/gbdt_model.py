import logging
import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

class GBDTModel:
    """
    Gradient Boosted Decision Tree model for credit risk / fraud detection.
    """

    def __init__(self):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.model = None
        self.feature_columns = None


    def prepare_dataset(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list
    ):
        """
        Prepare training dataset.
        """

        self.logger.info("Preparing dataset for GBDT model")
        data = df[feature_cols + [target_col]].copy()

        data = data.dropna()

        X = data[feature_cols]
        y = data[target_col]

        self.feature_columns = feature_cols

        return X, y


    def train(
        self,
        X,
        y,
        test_size: float = 0.2
    ):
        """
        Train GBDT model.
        """

        self.logger.info("Splitting dataset")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )
        self.logger.info("Training XGBoost model")

        self.model = xgb.XGBClassifier(

            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        self.model.fit(X_train, y_train)

        self.logger.info("Model training completed")

        preds = self.model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds)

        self.logger.info(f"AUC Score: {auc}")

        return auc
    def predict(self, X: pd.DataFrame):
        """
        Predict default probability.
        """

        if self.model is None:
            raise ValueError("Model not trained")

        self.logger.info("Generating predictions")

        probs = self.model.predict_proba(X)[:, 1]

        return probs
def predict_class(self, X: pd.DataFrame, threshold: float = 0.5):
        """
        Convert probabilities into class predictions.
        """

        probs = self.predict(X)

        return (probs >= threshold).astype(int)


def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance.
        """

        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.feature_importances_

        df = pd.DataFrame({

            "feature": self.feature_columns,
            "importance": importance

        }).sort_values("importance", ascending=False)

        return df
def save_model(self, path: str):
        """
        Save trained model.
        """

        self.logger.info(f"Saving model to {path}")

        self.model.save_model(path)


def load_model(self, path: str):
        """
        Load model.
        """

        self.logger.info(f"Loading model from {path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)