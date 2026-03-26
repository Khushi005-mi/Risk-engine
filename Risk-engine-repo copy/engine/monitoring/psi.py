import logging
import pandas as pd
import numpy as np

from typing import Dict, Any


class PSICalculator:
    """
    Population Stability Index calculator for drift detection.
    """

    def __init__(self, bins: int = 10):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.bins = bins


    def _create_bins(self, reference: pd.Series):
        """
        Create bins using quantiles from reference data.
        """

        quantiles = np.linspace(0, 1, self.bins + 1)

        bin_edges = np.quantile(reference, quantiles)

        # Remove duplicates (important edge case)
        bin_edges = np.unique(bin_edges)

        return bin_edges


    def _compute_distribution(
        self,
        data: pd.Series,
        bin_edges: np.ndarray
    ):
        """
        Compute normalized histogram distribution.
        """

        counts, _ = np.histogram(data, bins=bin_edges)

        distribution = counts / len(data)

        # Avoid division by zero
        distribution = np.where(distribution == 0, 1e-6, distribution)

        return distribution


    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Calculate PSI score for a single feature.
        """

        self.logger.info("Calculating PSI")

        reference = reference.dropna()
        current = current.dropna()

        bin_edges = self._create_bins(reference)

        expected = self._compute_distribution(reference, bin_edges)
        actual = self._compute_distribution(current, bin_edges)

        psi_values = (actual - expected) * np.log(actual / expected)

        psi_score = np.sum(psi_values)

        return float(psi_score)


    def calculate_psi_dataframe(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate PSI for all numeric features.
        """

        results = {}

        for col in reference_df.columns:

            if col not in current_df.columns:
                continue

            if pd.api.types.is_numeric_dtype(reference_df[col]):

                psi_score = self.calculate_psi(
                    reference_df[col],
                    current_df[col]
                )

                results[col] = {
                    "psi": psi_score,
                    "drift_level": self._interpret_psi(psi_score)
                }

        return results


    def _interpret_psi(self, psi: float) -> str:
        """
        Interpret PSI score.
        """

        if psi < 0.1:
            return "LOW"
        elif psi < 0.25:
            return "MEDIUM"
        else:
            return "HIGH"