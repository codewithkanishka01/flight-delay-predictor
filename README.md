# ✈️ Flight Delay Predictor

An end-to-end Machine Learning web application that predicts pre-departure flight arrival delays (>= 15 minutes) using an optimized XGBoost model with zero target-leakage features.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview
Flight delays cause billions of dollars in losses and operational inefficiencies annually. This project addresses the challenge by building a pre-departure prediction pipeline:
* **Target:** Predicting whether a flight arrival will be delayed by **15 or more minutes**.
* **Zero Target-Leakage:** Strictly uses features known prior to departure (Scheduled Departure Time, Scheduled Arrival Time, Carrier, Origin, Destination, Distance, Date).
* **Model:** Tuned XGBoost Classifier with an optimized decision threshold.

---

## 🚀 Live Demo & UI Features
* **Interactive Dashboard:** Select Flight Date, Airline Carrier, Origin & Destination Airports, and Departure/Arrival Times.
* **Instant Inference:** Real-time delay prediction with confidence/probability percentage.
* **Model Transparency:** Displays live evaluation metrics (Accuracy, ROC-AUC, F1-Score, Precision, Recall).

---

## 🛠️ Tech Stack
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
* **Backend:** Flask, Gunicorn (WSGI)
* **Frontend:** Responsive Glassmorphism UI (HTML5, CSS3, JavaScript)
* **Deployment:** Docker, Render, Hugging Face Spaces

---

## 💻 Local Setup & Execution

### 1. Clone the repository
```bash
git clone https://github.com/codewithkanishka01/flight-delay-predictor.git
cd flight-delay-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5001`.

---

## 🐳 Docker Deployment
```bash
# Build Docker image
docker build -t flight-delay-predictor .

# Run Docker container
docker run -p 5001:5001 flight-delay-predictor
```

---

## 👤 Author
**Kanishka Pandey**
* GitHub: [@codewithkanishka01](https://github.com/codewithkanishka01)
