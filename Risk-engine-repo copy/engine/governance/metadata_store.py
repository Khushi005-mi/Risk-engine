import json
import os
from typing import Dict, Any


class MetadataStore:
    def __init__(self, path: str = "governance/metadata.json"):
        self.path = path
        self._ensure()

    def _ensure(self):
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def log(self, data: Dict[str, Any]):
        with open(self.path, "r") as f:
            logs = json.load(f)

        logs.append(data)

        with open(self.path, "w") as f:
            json.dump(logs, f, indent=4)

    def fetch_all(self):
        with open(self.path, "r") as f:
            return json.load(f)
        