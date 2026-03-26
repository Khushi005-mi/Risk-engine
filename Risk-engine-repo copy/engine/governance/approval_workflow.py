class ApprovalWorkflow:
    def __init__(self, min_accuracy: float = 0.7, max_bias: float = 0.2):
        self.min_accuracy = min_accuracy
        self.max_bias = max_bias

    def evaluate(self, metrics: dict) -> bool:
        accuracy = metrics.get("accuracy", 0)
        bias = metrics.get("bias", 1)

        if accuracy >= self.min_accuracy and bias <= self.max_bias:
            return True

        return False

    def approve_or_reject(self, metrics: dict) -> str:
        if self.evaluate(metrics):
            return "approved"
        return "rejected"