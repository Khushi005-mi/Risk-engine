from pydantic import BaseModel, Field, validator
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Input schema for credit risk prediction
    """

    customer_id: int = Field(..., gt=0)
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    debt: float = Field(..., ge=0)
    employment_years: float = Field(..., ge=0)
    credit_history_length: float = Field(..., ge=0)

    @validator("income")
    def income_sanity(cls, v):
        if v > 1e9:
            raise ValueError("Income unrealistic")
        return v

    @validator("debt")
    def debt_vs_income(cls, v, values):
        income = values.get("income")
        if income and v > 5 * income:
            raise ValueError("Debt too high compared to income")
        return v


class PredictionResponse(BaseModel):
    """
    Output schema
    """

    customer_id: int
    risk_score: float
    default_probability: float
    risk_level: str