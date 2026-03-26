import pandas as pd
from typing import List, Dict


class RowValidator:
    """
    Performs row-level business rule validation.
    """

    def __init__(self):
        self.errors: List[Dict] = []

    def validate(self, df: pd.DataFrame):
        """
        Validate each row of the dataset.
        Returns:
            valid_df
            error_records
        """

        valid_rows = []

        for index, row in df.iterrows():

            row_errors = self._validate_row(row)

            if row_errors:
                for err in row_errors:
                    self.errors.append({
                        "row_index": index,
                        "error": err
                    })
            else:
                valid_rows.append(row)

        valid_df = pd.DataFrame(valid_rows)

        return valid_df, self.errors

    def _validate_row(self, row) -> List[str]:

        errors = []

        income = row.get("income")
        loan_amount = row.get("loan_amount")
        loan_term = row.get("loan_term")

        if income is not None and income <= 0:
            errors.append("income must be greater than 0")

        if loan_amount is not None and loan_amount <= 0:
            errors.append("loan_amount must be greater than 0")

        if loan_term is not None:
            if loan_term < 6 or loan_term > 360:
                errors.append("loan_term must be between 6 and 360 months")

        if income is not None and loan_amount is not None:
            if loan_amount > income * 10:
                errors.append("loan_amount exceeds allowable income multiplier")

        return errors