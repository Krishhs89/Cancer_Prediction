import pandas as pd
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from bedrock_features import generate_risk_score, get_bedrock_client
except ImportError:
    pass

def batch_predict(input_file, output_file):
    print(f"Starting batch processing for {input_file}...")
    
    # Load Artifacts
    try:
        model = joblib.load("../output/model.joblib")
        scaler = joblib.load("../output/scaler.joblib")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return

    # Load Data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records.")
    
    # Generate LLM Features if needed
    if 'pathology_notes' in df.columns and 'llm_risk_score' not in df.columns:
        print("Generating LLM features...")
        client = get_bedrock_client()
        df['llm_risk_score'] = df['pathology_notes'].apply(lambda x: generate_risk_score(x, client))
    elif 'llm_risk_score' not in df.columns:
        print("Warning: No pathology_notes or llm_risk_score found. Using default score 0.")
        df['llm_risk_score'] = 0
        
    # Prepare Features for Model
    # Ensure columns match model expectation
    feature_cols = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'llm_risk_score'
    ]
    
    # Check for missing columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}")
        return

    X = df[feature_cols]
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Save Results
    df['prediction'] = predictions
    df['malignancy_probability'] = probabilities
    
    df.to_csv(output_file, index=False)
    print(f"Batch processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage: python src/batch_inference.py ../data/new_patients.csv
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = input_path.replace(".csv", "_predictions.csv")
        batch_predict(input_path, output_path)
    else:
        print("Usage: python src/batch_inference.py <input_csv>")
