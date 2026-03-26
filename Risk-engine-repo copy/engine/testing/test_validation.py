import pytest
import pandas as pd

from engine.validation.batch_validator import BatchValidator


def test_valid_data_passes():

    df = pd.DataFrame({
        "customer_id": [1, 2],
        "age": [25, 40],
        "income": [50000, 80000]
    })

    validator = BatchValidator()

    result = validator.validate(df)

    assert result["valid"] is True


def test_missing_column_fails():

    df = pd.DataFrame({
        "customer_id": [1, 2],
        "age": [25, 40]
    })

    validator = BatchValidator()

    result = validator.validate(df)

    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_invalid_types():

    df = pd.DataFrame({
        "customer_id": ["A", "B"],
        "age": ["young", "old"],
        "income": [50000, 80000]
    })

    validator = BatchValidator()

    result = validator.validate(df)

    assert result["valid"] is False