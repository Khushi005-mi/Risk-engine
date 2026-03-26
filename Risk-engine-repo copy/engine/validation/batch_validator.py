import pandas as pd 
from typing import List, Dict
class BatchValidator:
    """
    Performs batch level validation checks across the dataset
    """
MAX_BATCH_SIZE = 100000
MIN_BATCH_SIZE = 1
def __init__(self):
    self.errors:List[Dict] = []
    self.warnings: List[Dict] = []

def validate(self, df:pd.DataFrame):
    """
    Run all batch-level validations.
    """
    self._check_batch_size(df)
    self._check_duplicate_customers(df)
    self._check_income_distribution(df)

    return self.errors, self.warnings

def _check_batch_size(self, df:pd.DataFrame):
    size = len(df)
    if size < self.MIN_BATCH_SIZE:
        self.errors.append({
            "error": "Batch contains no valid rows"
        })
    if size > self.MAX_BATCH_SIZE:
        self.errors.append({
            "error": f"Batch size exceed limit ({self.MAX_BATCH_SIZE})"
        })  
def _check_duplicate_customers(self,df:pd.DataFrame):
    if "income" not in df.columns:
        return
    duplicates = df[df.duplicated("customer_id")]

    if not duplicates.empty:
            self.warnings.append({
                "warning": "Duplicate customer_id detected",
                "count": len(duplicates)
            })

def _check_income_distribution(self, df: pd.DataFrame):

    if "income" not in df.columns:
            return

    mean_income = df["income"].mean() # we have taken mean of income then we will compare this with the dataset

    if mean_income > 5_000_000:
            self.warnings.append({
                "warning": "Income distribution unusually high",
                "mean_income": mean_income
            })

    if mean_income < 100:
            self.warnings.append({
                "warning": "Income distribution unusually low",
                "mean_income": mean_income
            })        



