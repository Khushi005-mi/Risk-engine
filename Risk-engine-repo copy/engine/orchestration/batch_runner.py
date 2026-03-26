import logging
import pandas as pd
from datetime import datetime

from engine.inference_pipeline import InferencePipeline


class BatchRunner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pipeline = InferencePipeline()

    def run_batch(self, input_path: str, output_path: str, model_name: str):

        self.logger.info(f"Processing batch: {input_path}")

        # 1. Load
        df = pd.read_csv(input_path)

        # 2. Run inference
        result = self.pipeline.run(df, model_name)

        # 3. Add metadata
        result["scoring_timestamp"] = datetime.utcnow().isoformat()

        # 4. Save output
        result.to_csv(output_path, index=False)

        self.logger.info(f"Batch completed: {output_path}")

        return output_path