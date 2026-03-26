import logging
import pandas as pd
import numpy as np

from typing import Dict, Any

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)


class PerformanceMonitor:
    """
    Monitors model performance in production.
    """

    def __init__(self, config: Dict[str, Any] = None):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.config = config or {}

        self.auc_threshold = self.config.get("auc_threshold", 0.65)
        self.f1_threshold = self.config.get("f1_threshold", 0.5)
def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute core performance metrics.
        """

        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            "auc": roc_auc_score(y_true, y_pred_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "brier_score": brier_score_loss(y_true, y_pred_proba)
        }

        return metrics
def segment_performance(
        self,
        df: pd.DataFrame,
        segment_col: str,
        target_col: str,
        pred_col: str
    ) -> Dict[str, Any]:
        """
        Evaluate performance across segments.
        """

        self.logger.info("Computing segment-wise performance")

        results = {}

        for segment in df[segment_col].unique():

            segment_df = df[df[segment_col] == segment]

            if len(segment_df) < 10:
                continue  # avoid noise

            metrics = self.compute_metrics(
                y_true=segment_df[target_col].values,
                y_pred_proba=segment_df[pred_col].values
            )

            results[str(segment)] = metrics

        return results
def detect_performance_drop(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect degradation in performance.
        """

        self.logger.info("Checking for performance degradation")

        drops = {}

        for metric in reference_metrics:

            ref = reference_metrics[metric]
            cur = current_metrics.get(metric, ref)

            change = cur - ref

            drops[metric] = {
                "reference": ref,
                "current": cur,
                "change": change
            }

        alert = (
            current_metrics["auc"] < self.auc_threshold or
            current_metrics["f1_score"] < self.f1_threshold
        ) 

        return {
            "metric_changes": drops,
            "performance_alert": alert
        }

def generate_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        reference_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Full performance monitoring report.
        """

        current_metrics = self.compute_metrics(y_true, y_pred_proba)

        degradation = self.detect_performance_drop(
            reference_metrics,
            current_metrics
        )

        report = {
            "current_metrics": current_metrics,
            "degradation_analysis": degradation
        }

        return report