import logging
import pandas as pd
import numpy as np

from typing import Dict, Any


class BiasAuditor:
    """
    Audits model predictions for bias across sensitive groups.
    """

    def __init__(self):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)


    def demographic_parity(
        self,
        df: pd.DataFrame,
        sensitive_col: str,
        prediction_col: str
    ) -> Dict[str, Any]:
        """
        Compare positive prediction rates across groups.
        """

        self.logger.info("Calculating Demographic Parity")

        group_rates = df.groupby(sensitive_col)[prediction_col].mean()

        return group_rates.to_dict()


    def disparate_impact(
        self,
        df: pd.DataFrame,
        sensitive_col: str,
        prediction_col: str
    ) -> float:
        """
        Compute Disparate Impact Ratio.
        """

        self.logger.info("Calculating Disparate Impact")

        group_rates = df.groupby(sensitive_col)[prediction_col].mean()

        min_rate = group_rates.min()
        max_rate = group_rates.max()

        if max_rate == 0:
            return 0.0

        ratio = min_rate / max_rate

        return float(ratio)


    def equal_opportunity(
        self,
        df: pd.DataFrame,
        sensitive_col: str,
        prediction_col: str,
        target_col: str
    ) -> Dict[str, Any]:
        """
        Compare true positive rates across groups.
        """

        self.logger.info("Calculating Equal Opportunity")

        results = {}

        groups = df[sensitive_col].unique()

        for group in groups:

            group_df = df[df[sensitive_col] == group]

            true_positive = (
                (group_df[prediction_col] == 1) &
                (group_df[target_col] == 1)
            ).sum()

            actual_positive = (group_df[target_col] == 1).sum()

            tpr = true_positive / actual_positive if actual_positive > 0 else 0

            results[group] = tpr

        return results


    def generate_bias_report(
        self,
        df: pd.DataFrame,
        sensitive_col: str,
        prediction_col: str,
        target_col: str
    ) -> Dict[str, Any]:
        """
        Full bias audit report.
        """

        dp = self.demographic_parity(df, sensitive_col, prediction_col)

        di = self.disparate_impact(df, sensitive_col, prediction_col)

        eo = self.equal_opportunity(df, sensitive_col, prediction_col, target_col)

        report = {
            "demographic_parity": dp,
            "disparate_impact": di,
            "equal_opportunity": eo,
            "bias_flag": self._flag_bias(di)
        }

        return report


    def _flag_bias(self, di: float) -> bool:
        """
        Flag bias using 80% rule.
        """

        return di < 0.8