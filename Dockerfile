FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# If requirements.txt doesn't exist, we install manually for this demo
RUN pip install pandas numpy scikit-learn fastapi uvicorn boto3 scipy joblib

# Copy source code
COPY src/ src/

# Copy pre-trained models (in a real scenario, these might be pulled from S3)
COPY output/ output/

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
