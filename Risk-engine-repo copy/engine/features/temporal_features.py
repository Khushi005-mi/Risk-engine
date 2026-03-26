import pandas as pd 

class TemporalFeatures:
    """
    Generates time-based credit risk features.
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = self._loan_term_years(df)
        df = self._payment_velocity(df)
        df = self._risk_duration_factor(df)

        return df
    def _loan_term_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert loan terms from months to years.
        """
        if "loan_term" in df.columns:
            df["loan_term_years"] = df['loan_terms'] / 12
        return df 
    def _payment_velocity(self, df: pd.DataFrame)  -> pd.DataFrame:
        """
        Measures payment pressure relative to loan duration """

        if "loan_amount" in df.columns and "loan_term" in df.columns:
            df["payment_velocity"]  =  df["loan_amount"] / (df["loan_term"] + 1 )

        return df
    def _risk_duration_factor(self, df: pd.DataFrame)  -> pd.DataFrame:
        """
        Proxy features capturing risk exposure duration. 
        """
        if "loan_term_years" in df.columns :
            df["risk_duration_factor"]  =   (df["loan_term_years"] + 1)
            
        return df      