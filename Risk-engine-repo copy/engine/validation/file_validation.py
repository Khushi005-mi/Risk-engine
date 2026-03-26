# This module ensures the input file itself is valid, before checking rows or schemas.
# It typically validates:
# File exists
# File format (CSV/JSON/Parquet etc.)
# File size
# Encoding
# Empty file
# Basic column presence
# Duplicate columns
# If this fails → the system stops immediately.
# This prevents:
# corrupted uploads
# empty files
# wrong formats
# malicious inputs
from pathlib import Path
import pandas as pd


class FileValidator:
    """
    Validates input files before schema and row validation.
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".json"}
    MAX_FILE_SIZE_MB = 100

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def validate(self):
        """
        Run all file-level validations.
        """
        self._check_exists()
        self._check_format()
        self._check_size()

        df = self._load_file()

        self._check_empty(df)
        self._check_duplicate_columns(df)

        return df

    def _check_exists(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def _check_format(self):
        if self.file_path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

    def _check_size(self):
        size_mb = self.file_path.stat().st_size / (1024 * 1024)

        if size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {size_mb:.2f}MB. "
                f"Max allowed: {self.MAX_FILE_SIZE_MB}MB"
            )

    def _load_file(self):
        """
        Load file into DataFrame.
        """

        if self.file_path.suffix == ".csv":
            return pd.read_csv(self.file_path)

        elif self.file_path.suffix == ".parquet":
            return pd.read_parquet(self.file_path)

        elif self.file_path.suffix == ".json":
            return pd.read_json(self.file_path)

        else:
            raise ValueError("Unsupported file type")

    def _check_empty(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Input file is empty")

    def _check_duplicate_columns(self, df: pd.DataFrame):
        duplicates = df.columns[df.columns.duplicated()].tolist()

        if duplicates:
            raise ValueError(f"Duplicate columns found: {duplicates}")