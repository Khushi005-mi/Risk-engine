import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


class SchemaV1Validator:
    """
    Validator for schema version v1.
    Enforces the data contract defined in data_contracts/schema_v1.json.
    """

    def __init__(self, schema_path: str = "data_contracts/schema_v1.json"):
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, "r") as f:
            return json.load(f)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all schema-level validations.
        """

        self._check_required_columns(df)
        self._check_column_types(df)
        self._check_non_nullable(df)
        self._check_categorical_values(df)
        self._check_ranges(df)

        return df

    def _check_required_columns(self, df: pd.DataFrame):
        required = set(self.schema.get("required_columns", []))
        present = set(df.columns)

        missing = required - present

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _check_column_types(self, df: pd.DataFrame):
        type_map = self.schema.get("column_types", {})

        for column, expected_type in type_map.items():

            if column not in df.columns:
                continue

            if expected_type == "int":
                if not pd.api.types.is_integer_dtype(df[column]):
                    raise TypeError(f"Column {column} must be integer")

            elif expected_type == "float":
                if not pd.api.types.is_numeric_dtype(df[column]):
                    raise TypeError(f"Column {column} must be numeric")

            elif expected_type == "category":
                if not pd.api.types.is_object_dtype(df[column]):
                    raise TypeError(f"Column {column} must be categorical")

    def _check_non_nullable(self, df: pd.DataFrame):
        non_nullable: List[str] = self.schema.get("non_nullable", [])

        for column in non_nullable:
            if df[column].isnull().any():
                raise ValueError(f"Column {column} contains null values")

    def _check_categorical_values(self, df: pd.DataFrame):
        categorical_rules = self.schema.get("categorical_values", {})

        for column, allowed_values in categorical_rules.items():

            if column not in df.columns:
                continue

            invalid = df[~df[column].isin(allowed_values)]

            if not invalid.empty:
                raise ValueError(
                    f"Column {column} contains invalid values: "
                    f"{invalid[column].unique().tolist()}"
                )

    def _check_ranges(self, df: pd.DataFrame):
        ranges = self.schema.get("ranges", {})

        for column, bounds in ranges.items():

            if column not in df.columns:
                continue

            min_val, max_val = bounds

            if ((df[column] < min_val) | (df[column] > max_val)).any():
                raise ValueError(
                    f"Column {column} contains values outside range "
                    f"[{min_val}, {max_val}]"
                )