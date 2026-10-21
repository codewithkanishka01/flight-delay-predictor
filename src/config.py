import os

# Base paths
WORKSPACE_DIR = "/Users/kanishkapandey/Desktop/ML"
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
MODEL_DIR = os.path.join(WORKSPACE_DIR, "models")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset paths
RAW_DATA_PATH = "/Users/kanishkapandey/Desktop/flights_sample_3m.csv"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_processed.parquet")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_processed.parquet")

# Randomized Search Configuration
TUNING_ROWS = 300000
RANDOM_STATE = 42

# Column specifications
ORIGINAL_COLS = [
    "FL_DATE",
    "AIRLINE",
    "AIRLINE_CODE",
    "FL_NUMBER",
    "ORIGIN",
    "ORIGIN_CITY",
    "DEST",
    "DEST_CITY",
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
    "ELAPSED_TIME",
    "AIR_TIME",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED"
]

CATEGORICAL_COLS = [
    "AIRLINE",
    "AIRLINE_CODE",
    "ORIGIN",
    "ORIGIN_CITY",
    "DEST",
    "DEST_CITY"
]

NUMERICAL_COLS = [
    "FL_NUMBER",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "CRS_DEP_HOUR",
    "CRS_DEP_MINUTE",
    "CRS_ARR_HOUR",
    "CRS_ARR_MINUTE"
]

MODEL_FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS
TARGET_COL = "DELAYED"
