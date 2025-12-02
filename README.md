# Cancer Prediction System (Production Ready)

Welcome to the **Cancer Prediction System**, an enterprise-grade MLOps solution for predicting tumor malignancy. This system integrates classical machine learning with Generative AI (Amazon Bedrock) to provide accurate, explainable diagnoses.

## 🚀 Key Features
- **Hybrid AI Engine**: Combines Random Forest (Structured Data) + Claude 3 (Unstructured Pathology Notes).
- **Production Ready**: Includes REST API, Batch Processing, and Docker support.
- **Interactive Dashboard**: PyShiny web app for real-time demos.
- **Self-Healing**: Automated Drift Detection and Retraining pipeline.
- **Explainable**: Provides Feature Importance and LLM Risk Scores.

## 📚 Documentation
- **[Architecture Workflow](project_workflow.md)**: End-to-End AWS MLOps diagram.
- **[API Documentation](api_documentation.md)**: Details on `/predict` and `/health` endpoints.
- **[Results Report](results_report.md)**: Model performance metrics and insights.

## 🛠️ Quick Start

### 1. Prerequisites
- Docker installed
- Python 3.9+ installed
- AWS Credentials configured (for Bedrock)

### 2. Run with Docker (API)
```bash
# Build the image
docker build -t cancer-prediction-api .

# Run the container
docker run -p 8000:8000 -e AWS_PROFILE=default cancer-prediction-api
```

### 3. Run Interactive Dashboard (PyShiny)
Visualize the model and interact with it using the web dashboard:
```bash
pip install shiny pandas scikit-learn joblib boto3
python -m shiny run src/dashboard.py
```

### 4. Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "mean_radius": 15.0,
           "mean_texture": 20.0,
           "mean_perimeter": 100.0,
           "mean_area": 700.0,
           "mean_smoothness": 0.1,
           "mean_compactness": 0.15,
           "mean_concavity": 0.2,
           "mean_concave_points": 0.1,
           "mean_symmetry": 0.2,
           "mean_fractal_dimension": 0.07,
           "pathology_note": "Large irregular mass."
         }'
```

## 📂 Project Structure
- `src/app.py`: **Online Inference API** (FastAPI).
- `src/dashboard.py`: **Interactive Dashboard** (PyShiny).
- `src/batch_inference.py`: **Batch Processing** script.
- `src/monitoring.py`: **Drift Detection** system.
- `src/bedrock_features.py`: **LLM Integration** (Amazon Bedrock).
- `Dockerfile`: Deployment configuration.
