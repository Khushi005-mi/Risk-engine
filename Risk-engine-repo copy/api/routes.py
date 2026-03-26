import logging
import pandas as pd

from fastapi import APIRouter, HTTPException

from api.request_models import PredictionRequest, PredictionResponse

from engine.validation.batch_validator import BatchValidator
from engine.features.feature_pipeline import FeaturePipeline
from engine.modeling.inference_model import InferenceModel

router = APIRouter()

logger = logging.getLogger(__name__)

validator = BatchValidator()
pipeline = FeaturePipeline()
model = InferenceModel()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:
        # Convert to DataFrame
        df = pd.DataFrame([request.dict()])

        # Step 1: Validate
        validation_result = validator.validate(df)

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=validation_result["errors"]
            )

        # Step 2: Feature Engineering
        features = pipeline.transform(df)

        # Step 3: Prediction
        proba = model.predict_proba(features.values)[0]

        # Step 4: Risk Level
        if proba > 0.8:
            risk_level = "HIGH"
        elif proba > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            customer_id=request.customer_id,
            risk_score=proba,
            default_probability=proba,
            risk_level=risk_level
        )

    except Exception as e:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )