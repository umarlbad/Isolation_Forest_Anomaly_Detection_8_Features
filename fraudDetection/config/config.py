from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logging"

FEATURE_COLS = [
    "hour_in_dec", "time_last_transaction", "amt_transaction_hourly", "amt_cumulative",
    "amt_diff_last_transaction", "amt_relative_transaction", "transaction_freq_daily", "transaction_freq_variability"]

IF_CONFIG = {
    "num_estimators": 200, "max_samples": 256, "max_features": 1.0,
    "contamination": 0.0128, "random_seed": 42}


SCORE_COL    = "anomalyScore"
LABEL_COL    = "predictedLabel"
FEAT_COL_STD = "scaledFeatures"