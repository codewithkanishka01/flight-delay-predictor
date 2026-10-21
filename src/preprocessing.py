import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def engineer_features_df(df_in):
    """
    Perform pre-departure feature engineering on the raw input DataFrame.
    This function is target-free and relies only on pre-flight schedules.
    """
    df = df_in.copy()
    
    # 1. Temporal features from FL_DATE
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%Y-%m-%d")
    df["YEAR"] = df["FL_DATE"].dt.year.astype(int)
    df["MONTH"] = df["FL_DATE"].dt.month.astype(int)
    df["DAY"] = df["FL_DATE"].dt.day.astype(int)
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek.astype(int)
    df = df.drop(columns=["FL_DATE"])
    
    # 2. Pre-departure times (CRS_DEP_TIME and CRS_ARR_TIME are integers in HHMM format)
    df["CRS_DEP_TIME"] = df["CRS_DEP_TIME"].fillna(0).astype(int)
    df["CRS_ARR_TIME"] = df["CRS_ARR_TIME"].fillna(0).astype(int)
    
    df["CRS_DEP_HOUR"] = df["CRS_DEP_TIME"] // 100
    df["CRS_DEP_MINUTE"] = df["CRS_DEP_TIME"] % 100
    
    df["CRS_ARR_HOUR"] = df["CRS_ARR_TIME"] // 100
    df["CRS_ARR_MINUTE"] = df["CRS_ARR_TIME"] % 100
    df = df.drop(columns=["CRS_DEP_TIME", "CRS_ARR_TIME"])
    
    # 3. Route feature
    df["route"] = df["ORIGIN"].astype(str) + "_" + df["DEST"].astype(str)
    
    # Interaction features for carrier hub congestion
    df["AIRLINE_ORIGIN"] = df["AIRLINE"].astype(str) + "_" + df["ORIGIN"].astype(str)
    df["AIRLINE_DEST"] = df["AIRLINE"].astype(str) + "_" + df["DEST"].astype(str)
    
    # 4. Departure hour bins (Legitimate pre-flight scheduling patterns)
    # Early Morning: 5-8, Morning: 8-12, Afternoon: 12-17, Evening: 17-21, Night: 21-5
    df["DEP_HOUR_BIN"] = pd.cut(
        df["CRS_DEP_HOUR"],
        bins=[-1, 5, 8, 12, 17, 21, 25],
        labels=["Night", "Early_Morning", "Morning", "Afternoon", "Evening", "Night_Late"]
    ).astype(str)
    df["DEP_HOUR_BIN"] = df["DEP_HOUR_BIN"].replace("Night_Late", "Night")
    
    return df

def get_preprocessing_pipeline():
    """
    Construct the Scikit-learn ColumnTransformer pipeline.
    """
    categorical_cols = ["AIRLINE", "ORIGIN", "DEST", "route", "AIRLINE_ORIGIN", "AIRLINE_DEST", "DEP_HOUR_BIN"]
    numerical_cols = [
        "CRS_ELAPSED_TIME", "DISTANCE",
        "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
        "CRS_DEP_HOUR", "CRS_DEP_MINUTE",
        "CRS_ARR_HOUR", "CRS_ARR_MINUTE"
    ]
    
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor
