#This file creates core numeric features directly from raw columns.
import numpy as np
import pandas as pd
class BaseFeatures:
    """
    Generates base financial features from raw input data."""
    def transform(self , df:pd.DataFrame) -> pd.DataFrame:
        """
        Create base features used by downstream feature modules>
        """
        df = df.copy()

        df = self._income_features(df)
        df = self._loan_features(df)
        return df
    
    def _income_features(self, df:pd.DataFrame) -> pd.DataFrame:
        if "loan_amount" in df.columns:
            df["loan_amount_log"] = np.log1p(df["loan_amount"])
        if "loan_term" in df.columns and "loan_amount" in df.columns:
           df["loan_per_month"] = df["loan_amount"] / df["loan_term"]
        return df 
            