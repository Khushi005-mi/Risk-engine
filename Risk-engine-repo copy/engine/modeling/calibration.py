import logging
import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

def __init__(self, method: str = "platt"):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.method = method
        self.calibrator = None


def fit_platt(self, model, X, y):
        """
        Platt Scaling using logistic regression calibration.
        """

        self.logger.info("Training Platt calibration model")

        self.calibrator = CalibratedClassifierCV(
            base_estimator=model,
            method="sigmoid",
            cv=5
        )

        self.calibrator.fit(X, y)

        self.logger.info("Platt calibration completed")
def fit_isotonic(self, y_pred, y_true):
        """
        Train isotonic regression calibration.
        """

        self.logger.info("Training isotonic calibration")

        self.calibrator = IsotonicRegression(
            out_of_bounds="clip"
        )

        self.calibrator.fit(y_pred, y_true)

        self.logger.info("Isotonic calibration completed")
def predict(self, X=None, y_pred=None):
        """
        Generate calibrated probabilities.
        """

        if self.calibrator is None:
            raise ValueError("Calibrator not trained")

        if self.method == "platt":

            probs = self.calibrator.predict_proba(X)[:, 1]

        else:

            probs = self.calibrator.predict(y_pred)

        return probs

def evaluate(self, y_true, y_pred):
        """
        Evaluate calibration quality using Brier Score.
        """

        score = brier_score_loss(y_true, y_pred)

        self.logger.info(f"Brier Score: {score}")

        return score