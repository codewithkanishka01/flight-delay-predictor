FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and required assets
COPY app.py .
COPY static/ static/
COPY templates/ templates/
COPY models/xgb_model_tuned.json models/
COPY models/mappings.json models/
COPY models/threshold_config_10.json models/
COPY outputs/final_test_10_results.json outputs/

EXPOSE 5001

CMD exec gunicorn --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120 app:app
