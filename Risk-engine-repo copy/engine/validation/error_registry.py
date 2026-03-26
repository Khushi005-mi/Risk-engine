from pathlib import Path
import yaml


class ErrorRegistry:
    """
    Central registry for validation error codes and messages.
    Loads definitions from data_contracts/error_registry.yaml.
    """

    def __init__(self, registry_path: str = "data_contracts/error_registry.yaml"):
        self.registry_path = Path(registry_path)
        self.errors = self._load_registry()

    def _load_registry(self):
    
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Error registry file not found: {self.registry_path}"
            )

        with open(self.registry_path, "r") as f:
            return yaml.safe_load(f)

    def get_error(self, error_code: str):
        """
        Retrieve error metadata by code.
        """

        if error_code not in self.errors:
            raise ValueError(f"Unknown error code: {error_code}")

        return self.errors[error_code]

    def format_error(self, error_code: str, context: dict = None):
        """
        Format a structured error response.
        """

        error_def = self.get_error(error_code)

        error_message = error_def["message"]

        if context:
            error_message = error_message.format(**context)

        return {
            "error_code": error_code,
            "message": error_message,
            "severity": error_def.get("severity", "error"),
        }