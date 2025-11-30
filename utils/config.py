import os

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = os.path.join(ROOT_DIR, "data")
LOG_DIR   = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

RAW_DATA_DIR       = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR   = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_METRICS_DIR  = os.path.join(MODEL_DIR, "reports")

# Ensure folders exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_METRICS_DIR, exist_ok=True)