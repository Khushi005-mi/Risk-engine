import pandas as pd 
class BehaviouralFeatures:
    """
    Generates behavioral risk indicators from financial features.
    """

    def transform(self , df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self._repayment_capacity(df)
        df = self._loan_stress(df)
        df = self._income_stability_proxy(df)

        return df
    def _repayment_capacity(self , df:pd.DataFrame) -> pd.DataFrame:
        """
        Estimate borrower's ability to repay monthly loan.
        """
        if  "income_monthly" in df.column and "loan_per_month" in df.columns:
            df["loan_stress_score"] = df["loan_per_month"] / (df["income_monthly"] +1)
        return df
    def loan_stress(self , df: pd.DataFrame) -> pd.DataFrame:
        """
        calculate financial stress indicator.
        """
        if  "loan_per_month" in df.column and "income_monthly" in df.columns:
            df["loan_stress_score"] = df["loan_per_month"] / (df["income_monthly"] +1)
        return df
    def _income_stability_proxy(self , df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy feature estimating income stability.
        """
        if  "income" in df.column:
            df["income_stability_proxy"] = df["income"] / (df["income"].mean() +1)
        return df

           