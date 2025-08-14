# ~/Projects/sensor_ai/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "sensor_data"
RESULTS = PROJECT_ROOT / "results"
MODELS = RESULTS / "models"