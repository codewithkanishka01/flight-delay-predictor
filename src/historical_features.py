import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class HistoricalFeatureEncoder:
    """
    Computes leakage-free, smoothed historical delay rates and flight counts.
    Uses Out-of-Fold (OOF) mapping on the training set and maps raw training statistics
    to validation/test splits.
    """
    def __init__(self, alpha=25, n_splits=5, random_state=42):
        self.alpha = alpha
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_rate = 0.0
        self.group_stats = {}
        
        # Define groupings for historical statistics
        self.groupings = {
            "airline": ["AIRLINE"],
            "origin": ["ORIGIN"],
            "dest": ["DEST"],
            "route": ["ORIGIN", "DEST"],
            "airline_origin": ["AIRLINE", "ORIGIN"],
            "airline_dest": ["AIRLINE", "DEST"],
            "route_dep_hour": ["ORIGIN", "DEST", "CRS_DEP_HOUR"],
            "airline_dep_hour": ["AIRLINE", "CRS_DEP_HOUR"],
            "origin_dep_hour": ["ORIGIN", "CRS_DEP_HOUR"],
            "dest_dep_hour": ["DEST", "CRS_DEP_HOUR"]
        }

    def _get_key_series(self, df, cols):
        if len(cols) == 1:
            return df[cols[0]].astype(str)
        else:
            return df[cols].astype(str).agg("_".join, axis=1)

    def fit(self, X, y):
        """
        Fits the historical statistics on the training partition.
        """
        df = X.copy()
        df["target"] = y
        self.global_rate = y.mean()
        
        # Calculate statistics for each grouping
        for name, cols in self.groupings.items():
            keys = self._get_key_series(df, cols)
            
            # Count and sum of targets
            stats = df.groupby(keys)["target"].agg(["count", "sum"])
            
            # Smooth rate: (sum + alpha * global_rate) / (count + alpha)
            stats["smoothed_rate"] = (stats["sum"] + self.alpha * self.global_rate) / (stats["count"] + self.alpha)
            
            self.group_stats[name] = stats[["count", "smoothed_rate"]].to_dict(orient="index")
            
        return self

    def transform_train(self, X, y):
        """
        Applies Out-of-Fold (OOF) target encoding to the training set to prevent leakage.
        """
        df = X.copy()
        df["target"] = y
        
        # Initialize output dataframe
        X_out = pd.DataFrame(index=X.index)
        
        # Compute out-of-fold mappings
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Initialize arrays for OOF rates
        oof_rates = {name: np.zeros(len(df)) for name in self.groupings}
        oof_counts = {name: np.zeros(len(df)) for name in self.groupings}
        
        # Run K-Fold loops
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            fold_train = df.iloc[train_idx]
            fold_val = df.iloc[val_idx]
            
            fold_global_rate = fold_train["target"].mean()
            
            for name, cols in self.groupings.items():
                train_keys = self._get_key_series(fold_train, cols)
                val_keys = self._get_key_series(fold_val, cols)
                
                # Fit statistics on fold training data
                stats = fold_train.groupby(train_keys)["target"].agg(["count", "sum"])
                stats["smoothed_rate"] = (stats["sum"] + self.alpha * fold_global_rate) / (stats["count"] + self.alpha)
                
                # Map to validation indices
                mapped_stats = stats.reindex(val_keys)
                
                # Fill missing with global statistics
                rates = mapped_stats["smoothed_rate"].fillna(fold_global_rate).values
                counts = mapped_stats["count"].fillna(0).values
                
                # Assign to out arrays
                oof_rates[name][val_idx] = rates
                oof_counts[name][val_idx] = counts
                
        # Assign rates to output features
        for name in self.groupings:
            X_out[f"hist_delay_rate_{name}"] = oof_rates[name]
            
        # Add flight count features (using full training flight counts)
        for name, cols in self.groupings.items():
            if name in ["route", "airline", "origin"]:
                keys = self._get_key_series(df, cols)
                counts = keys.map(lambda k: self.group_stats[name].get(k, {}).get("count", 0))
                X_out[f"hist_flight_count_{name}"] = counts.fillna(0)
                
        return X_out

    def transform_val_test(self, X):
        """
        Maps training historical statistics directly to validation or test splits.
        """
        df = X.copy()
        X_out = pd.DataFrame(index=X.index)
        
        # Map group stats
        for name, cols in self.groupings.items():
            keys = self._get_key_series(df, cols)
            
            rates = keys.map(lambda k: self.group_stats[name].get(k, {}).get("smoothed_rate", self.global_rate))
            counts = keys.map(lambda k: self.group_stats[name].get(k, {}).get("count", 0))
            
            X_out[f"hist_delay_rate_{name}"] = rates.fillna(self.global_rate)
            
            if name in ["route", "airline", "origin"]:
                X_out[f"hist_flight_count_{name}"] = counts.fillna(0)
                
        return X_out
