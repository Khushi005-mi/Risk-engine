import logging
import pandas as pd
import numpy as np

from typing import Dict, Any
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
class DriftDetector:
    """
    Detects data drift and prediction drift in production.
    """

    def __init__(self, config: Dict[str, Any]):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.config = config

        self.num_threshold = config.get("num_drift_threshold", 0.05)
        self.cat_threshold = config.get("cat_drift_threshold", 0.05)

        
        def detect_numeric_drift(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Dict[str, Any]:
            """
        Detect drift using Kolmogorov-Smirnov test.
        """

        stat, p_value = ks_2samp(reference, current)

        drift = p_value < self.num_threshold

        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": drift
        }
    def detect_categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect drift using Chi-Square test.
        """

        ref_counts = reference.value_counts(normalize=True)
        cur_counts = current.value_counts(normalize=True)

        categories = list(set(ref_counts.index).union(set(cur_counts.index)))

        ref_vals = [ref_counts.get(cat, 0) for cat in categories]
        cur_vals = [cur_counts.get(cat, 0) for cat in categories]

        contingency = np.array([ref_vals, cur_vals])

        stat, p_value, _, _ = chi2_contingency(contingency)

        drift = p_value < self.cat_threshold

        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": drift
        }
def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect drift for all features.
        """

        results = {}

        for col in reference_df.columns:

            if col not in current_df.columns:
                continue

            if pd.api.types.is_numeric_dtype(reference_df[col]):

                result = self.detect_numeric_drift(
                    reference_df[col].dropna(),
                    current_df[col].dropna()
                )
def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect drift for all features.
        """

        results = {}

        for col in reference_df.columns:

            if col not in current_df.columns:
                continue

            if pd.api.types.is_numeric_dtype(reference_df[col]):

                result = self.detect_numeric_drift(
                    reference_df[col].dropna(),
                    current_df[col].dropna()
                )

