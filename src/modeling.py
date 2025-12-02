import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def train_and_evaluate():
    # Load data
    # Try loading the LLM-enhanced data first, fallback to original
    import os
    if os.path.exists("../data/cancer_data_with_llm.csv"):
        print("Loading data with LLM features...")
        df = pd.read_csv("../data/cancer_data_with_llm.csv")
    else:
        print("Loading original data (no LLM features)...")
        df = pd.read_csv("../data/cancer_data.csv")
    
    # Drop diagnosis (target) and pathology_notes (text, not used directly in model)
    drop_cols = ['diagnosis']
    if 'pathology_notes' in df.columns:
        drop_cols.append('pathology_notes')
        
    X = df.drop(drop_cols, axis=1)
    y = df['diagnosis']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        print(f"--- {name} Results ---")
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "auc": auc
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"../output/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()
        
    # Feature Importance (Random Forest)
    rf_model = results["Random Forest"]["model"]
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig("../output/feature_importance.png")
    plt.close()
    
    # Save Model and Scaler for Production
    import joblib
    print("Saving artifacts...")
    joblib.dump(rf_model, "../output/model.joblib")
    joblib.dump(scaler, "../output/scaler.joblib")
    print("Model and Scaler saved to ../output/")

    print("\nModeling complete. Results saved to ../output/")

if __name__ == "__main__":
    train_and_evaluate()
