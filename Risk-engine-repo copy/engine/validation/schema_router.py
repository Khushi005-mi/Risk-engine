from typing import Dict
import pandas as pd

from .schema_v1 import SchemaV1Validator


class SchemaRouter:
    """
    Routes incoming data to the correct schema validator
    based on schema version.
    """

    def __init__(self):
        self.schema_registry: Dict[str, object] = {
            "v1": SchemaV1Validator(),
        }

    def detect_schema_version(self, df: pd.DataFrame) -> str:
        """
        Detect schema version from the input data.

        Strategy:
        - Check for explicit column 'schema_version'
        - Otherwise default to v1
        """

        if "schema_version" in df.columns:
            return str(df["schema_version"].iloc[0])

        return "v1"

    def get_validator(self, version: str):
        """
        Return the correct schema validator.
        """

        if version not in self.schema_registry:
            raise ValueError(f"Unsupported schema version: {version}")

        return self.schema_registry[version]

    def validate(self, df: pd.DataFrame):
        """
        Main entrypoint.
        Detect schema and run validation.
        """

        version = self.detect_schema_version(df)

        validator = self.get_validator(version)

        return validator.validate(df)