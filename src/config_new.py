import os

# Base paths
WORKSPACE_DIR = "/Users/kanishkapandey/Desktop/ML"
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
MODEL_DIR = os.path.join(WORKSPACE_DIR, "models")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Dataset paths
RAW_DATA_PATH = "/Users/kanishkapandey/Desktop/flights_sample_3m.csv"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_leakage_free.parquet")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_leakage_free.parquet")
TEMPORAL_TRAIN_PATH = os.path.join(DATA_DIR, "temporal_train.parquet")
TEMPORAL_TEST_PATH = os.path.join(DATA_DIR, "temporal_test.parquet")

# Pre-departure candidate features (8 columns)
ORIGINAL_COLS = [
    "FL_DATE",
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE"
]

LEAKAGE_COLS = [
    "DEP_TIME",
    "DEP_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "ARR_TIME",
    "ARR_DELAY",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "AIRLINE_DOT",
    "AIRLINE_CODE",
    "DOT_CODE",
    "FL_NUMBER",
    "ORIGIN_CITY",
    "DEST_CITY"
]

TARGET_COL = "DELAYED"
RANDOM_STATE = 42
