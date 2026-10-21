import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Base paths (dynamically resolved for local & cloud deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Load mappings
mappings_path = os.path.join(MODEL_DIR, "mappings.json")
if os.path.exists(mappings_path):
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
else:
    mappings = {"airlines": [], "airline_to_code": {}, "airports": [], "airport_to_city": {}}

# Load final model (xgb_model_tuned.json)
model_path = os.path.join(MODEL_DIR, "xgb_model_tuned.json")
model = None
if os.path.exists(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)

# Load threshold config (threshold_config_10.json)
threshold_path = os.path.join(MODEL_DIR, "threshold_config_10.json")
locked_threshold = 0.50
if os.path.exists(threshold_path):
    with open(threshold_path, 'r') as f:
        t_config = json.load(f)
        locked_threshold = t_config.get("best_threshold_accuracy", 0.50)

# Load final evaluation metrics (final_test_10_results.json)
metrics_path = os.path.join(OUTPUT_DIR, "final_test_10_results.json")
final_metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        raw_metrics = json.load(f)
        final_metrics = {
            "Accuracy": raw_metrics.get("final_test_accuracy", 0.0),
            "Precision": raw_metrics.get("final_test_precision", 0.0),
            "Recall": raw_metrics.get("final_test_recall", 0.0),
            "ROC_AUC": raw_metrics.get("final_test_roc_auc", 0.0),
            "F1": raw_metrics.get("final_test_f1", 0.0)
        }

# Define expected features (exactly 15 engineered features matching locked 10 columns)
expected_features = [
    "AIRLINE", "AIRLINE_CODE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY",
    "DISTANCE", "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
    "CRS_DEP_HOUR", "CRS_DEP_MINUTE", "CRS_ARR_HOUR", "CRS_ARR_MINUTE"
]

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        airlines=mappings["airlines"],
        airports=mappings["airports"],
        threshold=locked_threshold,
        metrics=final_metrics,
        model_loaded=(model is not None)
    )

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Train the model first."}), 500
        
    try:
        # Get raw form values
        fl_date_str = request.form.get("fl_date") # YYYY-MM-DD
        airline = request.form.get("airline")
        origin = request.form.get("origin")
        dest = request.form.get("dest")
        crs_dep_time_str = request.form.get("crs_dep_time") # HH:MM
        crs_arr_time_str = request.form.get("crs_arr_time") # HH:MM
        distance = float(request.form.get("distance"))
        
        # 1. Resolve derived categorical features
        airline_code = mappings["airline_to_code"].get(airline, "")
        origin_city = mappings["airport_to_city"].get(origin, "")
        dest_city = mappings["airport_to_city"].get(dest, "")
        
        # 2. Engineer temporal features
        fl_date = datetime.strptime(fl_date_str, "%Y-%m-%d")
        year = fl_date.year
        month = fl_date.month
        day = fl_date.day
        day_of_week = fl_date.weekday() # Monday=0, Sunday=6
        
        dep_time = datetime.strptime(crs_dep_time_str, "%H:%M")
        crs_dep_hour = dep_time.hour
        crs_dep_minute = dep_time.minute
        
        arr_time = datetime.strptime(crs_arr_time_str, "%H:%M")
        crs_arr_hour = arr_time.hour
        crs_arr_minute = arr_time.minute
        
        # 3. Create DataFrame matching training format
        input_data = {
            "AIRLINE": [airline],
            "AIRLINE_CODE": [airline_code],
            "ORIGIN": [origin],
            "ORIGIN_CITY": [origin_city],
            "DEST": [dest],
            "DEST_CITY": [dest_city],
            "DISTANCE": [distance],
            "YEAR": [year],
            "MONTH": [month],
            "DAY": [day],
            "DAY_OF_WEEK": [day_of_week],
            "CRS_DEP_HOUR": [crs_dep_hour],
            "CRS_DEP_MINUTE": [crs_dep_minute],
            "CRS_ARR_HOUR": [crs_arr_hour],
            "CRS_ARR_MINUTE": [crs_arr_minute]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # 4. Cast categoricals
        categorical_cols = ["AIRLINE", "AIRLINE_CODE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY"]
        for col in categorical_cols:
            input_df[col] = input_df[col].astype(str).astype("category")
            
        # Ensure correct column order
        input_df = input_df[expected_features]
        
        # 5. Predict probability
        prob = float(model.predict_proba(input_df)[0, 1])
        prediction = 1 if prob >= locked_threshold else 0
        
        result = {
            "success": True,
            "prediction": prediction,
            "probability": f"{prob * 100:.2f}%",
            "threshold": f"{locked_threshold * 100:.2f}%",
            "airline_code": airline_code,
            "origin_city": origin_city,
            "dest_city": dest_city,
            "delayed": "SEVERE DELAY (At least 45 min late)" if prediction == 1 else "ON TIME / MINIMAL DELAY (< 45 min)"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
