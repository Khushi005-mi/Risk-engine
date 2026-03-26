from typing import Dict, Any, Callable


class ReplayEngine:
    def __init__(self, model_loader: Callable):
        """
        model_loader: function that takes path and returns model
        """
        self.model_loader = model_loader

    def replay(self, model_path: str, input_data: Dict[str, Any]):
        model = self.model_loader(model_path)

        prediction = model.predict([list(input_data.values())])

        return {
            "input": input_data,
            "prediction": prediction[0],
            "model_path": model_path
        }
    