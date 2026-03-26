import os
import json
from datetime import datetime
from typing import Dict, Any, List


class ModelRegistry:
    def __init__(self, registry_path: str = "governance/registry.json"):
        self.registry_path = registry_path
        self._ensure_registry()

    def _ensure_registry(self):
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump([], f)

    def _load(self) -> List[Dict]:
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save(self, data: List[Dict]):
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=4)

    def register_model(self, model_name: str, version: str, metrics: Dict[str, Any], path: str):
        registry = self._load()

        entry = {
            "model_name": model_name,
            "version": version,
            "metrics": metrics,
            "path": path,
            "status": "staging",
            "created_at": datetime.utcnow().isoformat()
        }

        registry.append(entry)
        self._save(registry)

        return entry

    def promote_to_production(self, model_name: str, version: str):
        registry = self._load()

        for model in registry:
            if model["model_name"] == model_name:
                model["status"] = "deprecated"

        for model in registry:
            if model["model_name"] == model_name and model["version"] == version:
                model["status"] = "production"

        self._save(registry)

    def get_production_model(self, model_name: str) -> Dict:
        registry = self._load()

        for model in registry:
            if model["model_name"] == model_name and model["status"] == "production":
                return model

        raise ValueError("No production model found")

    def list_models(self):
        return self._load()