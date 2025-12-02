import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modeling import train_and_evaluate

def check_drift_and_retrain(reference_data_path, new_data_path, threshold=0.05):
    print("Checking for data drift...")
    
    try:
        ref_df = pd.read_csv(reference_data_path)
        new_df = pd.read_csv(new_data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Features to monitor
    features = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 
        'llm_risk_score'
    ]
    
    drift_detected = False
    
    for feature in features:
        if feature in ref_df.columns and feature in new_df.columns:
            # KS Test
            stat, p_value = ks_2samp(ref_df[feature], new_df[feature])
            
            if p_value < threshold:
                print(f"DRIFT DETECTED in {feature} (p-value: {p_value:.4f})")
                drift_detected = True
            else:
                print(f"No drift in {feature} (p-value: {p_value:.4f})")
                
    if drift_detected:
        print("\n!!! Data Drift Detected !!! Triggering Retraining Pipeline...")
        # In a real system, we might merge new data with old data here
        # For simulation, we just re-run the training script
        train_and_evaluate()
        print("Retraining complete. New model deployed.")
    else:
        print("\nSystem healthy. No retraining needed.")

if __name__ == "__main__":
    # Example: python src/monitoring.py ../data/cancer_data_with_llm.csv ../data/new_batch_data.csv
    # For demo, we can just compare the training data to itself (no drift) or a modified version
    
    ref_path = "../data/cancer_data_with_llm.csv"
    
    # Create a dummy "drifted" file for testing if not provided
    drift_path = "../data/drifted_data.csv"
    if not os.path.exists(drift_path) and os.path.exists(ref_path):
        df = pd.read_csv(ref_path)
        df['mean_radius'] = df['mean_radius'] + 5 # Simulate drift
        df.to_csv(drift_path, index=False)
        
    check_drift_and_retrain(ref_path, drift_path)
