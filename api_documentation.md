# Cancer Prediction API Documentation

The **Cancer Prediction Service** provides a RESTful API for real-time malignancy risk assessment.

## Base URL
`http://localhost:8000` (Local) or `https://<your-domain>` (Production)

## Endpoints

### 1. Health Check
**GET** `/health`

Checks if the service is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Predict Diagnosis
**POST** `/predict`

Analyzes patient tumor metrics and pathology notes to predict malignancy.

**Request Body:**
```json
{
  "mean_radius": 14.2,
  "mean_texture": 19.5,
  "mean_perimeter": 92.1,
  "mean_area": 650.0,
  "mean_smoothness": 0.09,
  "mean_compactness": 0.12,
  "mean_concavity": 0.15,
  "mean_concave_points": 0.08,
  "mean_symmetry": 0.18,
  "mean_fractal_dimension": 0.06,
  "pathology_note": "Irregular mass with spiculated margins observed."
}
```

**Response:**
```json
{
  "diagnosis": "Malignant",
  "malignancy_probability": 0.87,
  "llm_risk_score": 8
}
```

**Field Descriptions:**
- `diagnosis`: The predicted class (Benign/Malignant).
- `malignancy_probability`: Confidence score (0.0 - 1.0).
- `llm_risk_score`: Risk score (1-10) extracted from the `pathology_note` by Amazon Bedrock.
