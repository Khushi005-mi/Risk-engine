import json


class FeatureListManager:

    def __init__(self, path: str):
        self.path = path
        self.feature_list = self._load()

    def _load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def validate_and_align(self, df):

        df_cols = list(df.columns)

        # 1. Missing check
        missing = set(self.feature_list) - set(df_cols)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # 2. Extra check
        extra = set(df_cols) - set(self.feature_list)
        if extra:
            raise ValueError(f"Unexpected features: {extra}")

        # 3. Order enforcement
        df = df[self.feature_list]

        return df