import logging
from typing import Dict, Any, List


class ExplanationFormatter:
    """
    Formats SHAP explanations into business-readable outputs.
    """

    def __init__(self):

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)


    def split_contributions(
        self,
        contributions: List[Dict[str, Any]]
    ):
        """
        Split features into risk drivers and protective factors.
        """

        risk_drivers = []
        protective_factors = []

        for item in contributions:

            if item["contribution"] > 0:
                risk_drivers.append(item)
            else:
                protective_factors.append(item)

        return risk_drivers, protective_factors


    def get_top_factors(
        self,
        factors: List[Dict[str, Any]],
        top_n: int = 3
    ):
        """
        Get top N factors based on contribution magnitude.
        """

        return sorted(
            factors,
            key=lambda x: abs(x["contribution"]),
            reverse=True
        )[:top_n]


    def format_feature_name(self, feature: str) -> str:
        """
        Convert feature names into human-readable form.
        """

        return feature.replace("_", " ").title()


    def build_summary(
        self,
        pd_score: float,
        risk_factors: List[Dict[str, Any]],
        protective_factors: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable explanation summary.
        """

        risk_names = [self.format_feature_name(f["feature"]) for f in risk_factors]
        protective_names = [self.format_feature_name(f["feature"]) for f in protective_factors]

        summary = f"Risk score is {round(pd_score, 2)}. "

        if risk_names:
            summary += f"Main risk drivers: {', '.join(risk_names)}. "

        if protective_names:
            summary += f"Protective factors: {', '.join(protective_names)}."

        return summary


    def format_explanation(
        self,
        shap_output: Dict[str, Any],
        pd_score: float,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Full formatting pipeline.
        """

        contributions = shap_output["prediction_contributions"]

        risk_drivers, protective_factors = self.split_contributions(contributions)

        top_risk = self.get_top_factors(risk_drivers, top_n)
        top_protective = self.get_top_factors(protective_factors, top_n)

        summary = self.build_summary(pd_score, top_risk, top_protective)

        formatted = {
            "pd_score": round(pd_score, 4),
            "risk_level": self._risk_band(pd_score),
            "top_risk_drivers": [
                {
                    "feature": self.format_feature_name(f["feature"]),
                    "impact": round(f["contribution"], 4)
                }
                for f in top_risk
            ],
            "top_protective_factors": [
                {
                    "feature": self.format_feature_name(f["feature"]),
                    "impact": round(f["contribution"], 4)
                }
                for f in top_protective
            ],
            "summary": summary
        }

        return formatted


    def _risk_band(self, pd_score: float) -> str:
        """
        Convert PD into risk category.
        """

        if pd_score < 0.2:
            return "LOW"
        elif pd_score < 0.5:
            return "MEDIUM"
        else:
            return "HIGH"