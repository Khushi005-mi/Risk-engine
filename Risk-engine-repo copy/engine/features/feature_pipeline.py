# Feature_pipeline.py

import pandas as pd

from validation.batch_validator import validate_batch
from validation.error_registry import log_error

from temporal_features import generate_temporal_features
from ratio_features import generate_ratio_features
from features.behavioural_features import generate_behavioural_features


class FeaturePipeline:

    def __init__(self):
        pass


    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full feature engineering pipeline.
        """

        try:

            # 1️⃣ Validate input batch
            validate_batch(df)

            # 2️⃣ Temporal features
            df = generate_temporal_features(df)

            # 3️⃣ Ratio features
            df = generate_ratio_features(df)

            # 4️⃣ Behavioural features
            df = generate_behavioural_features(df)

            return df

        except Exception as e:

            log_error(
                module="FeaturePipeline",
                message=str(e)
            )

            raise e