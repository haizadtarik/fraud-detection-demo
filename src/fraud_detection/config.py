from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_params(path: str = "conf/parameters.yml") -> dict:
    with open(PROJECT_ROOT / path) as f:
        return yaml.safe_load(f)