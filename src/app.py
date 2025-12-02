from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add src to path to import bedrock_features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from bedrock_features import generate_risk_score, get_bedrock_client
except ImportError:
    # Fallback if running from a different directory context
    pass

app = FastAPI(title="Cancer Prediction API", version="1.0")

# Load Artifacts
MODEL_PATH = "../output/model.joblib"
SCALER_PATH = "../output/scaler.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model = None
    scaler = None

# Initialize Bedrock Client
bedrock_client = None
try:
    # Only init if we can import the module
    from bedrock_features import get_bedrock_client
    bedrock_client = get_bedrock_client()
except:
    pass

class PatientData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    pathology_note: str = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PatientData):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Prepare Features
    features = {
        'mean_radius': data.mean_radius,
        'mean_texture': data.mean_texture,
        'mean_perimeter': data.mean_perimeter,
        'mean_area': data.mean_area,
        'mean_smoothness': data.mean_smoothness,
        'mean_compactness': data.mean_compactness,
        'mean_concavity': data.mean_concavity,
        'mean_concave_points': data.mean_concave_points,
        'mean_symmetry': data.mean_symmetry,
        'mean_fractal_dimension': data.mean_fractal_dimension
    }
    
    # 2. Get LLM Risk Score (if note provided)
    llm_score = 0
    if data.pathology_note:
        try:
            # Import here to avoid top-level failure if file missing
            from bedrock_features import generate_risk_score
            llm_score = generate_risk_score(data.pathology_note, bedrock_client)
        except Exception as e:
            print(f"LLM Error: {e}")
            llm_score = 5 # Default fallback
            
    features['llm_risk_score'] = llm_score
    
    # 3. Create DataFrame
    df = pd.DataFrame([features])
    
    # 4. Scale
    X_scaled = scaler.transform(df)
    
    # 5. Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    return {
        "diagnosis": "Malignant" if prediction == 1 else "Benign",
        "malignancy_probability": float(probability),
        "llm_risk_score": llm_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
