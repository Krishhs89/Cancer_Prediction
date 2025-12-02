import random
import csv
import os

def generate_cancer_data(n_samples=200):
    """
    Generates a synthetic dataset for cancer prediction using pure Python.
    """
    random.seed(42)
    
    # Feature names
    feature_names = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'diagnosis'
    ]
    
    data = []
    
    for _ in range(n_samples):
        # 0: Benign, 1: Malignant
        diagnosis = 1 if random.random() < 0.37 else 0
        
        row = []
        # Generate features based on diagnosis
        if diagnosis == 0:
            # Benign: smaller, more regular
            row.append(random.gauss(10, 2)) # radius
            row.append(random.gauss(15, 3)) # texture
            row.append(random.gauss(70, 10)) # perimeter
            row.append(random.gauss(400, 100)) # area
            row.append(random.gauss(0.08, 0.01)) # smoothness
            row.append(random.gauss(0.05, 0.02)) # compactness
            row.append(random.gauss(0.03, 0.02)) # concavity
            row.append(random.gauss(0.02, 0.01)) # concave points
            row.append(random.gauss(0.18, 0.02)) # symmetry
            row.append(random.gauss(0.06, 0.01)) # fractal dimension
        else:
            # Malignant: larger, irregular
            row.append(random.gauss(15, 3)) # radius
            row.append(random.gauss(20, 4)) # texture
            row.append(random.gauss(100, 20)) # perimeter
            row.append(random.gauss(800, 200)) # area
            row.append(random.gauss(0.10, 0.02)) # smoothness
            row.append(random.gauss(0.12, 0.04)) # compactness
            row.append(random.gauss(0.15, 0.05)) # concavity
            row.append(random.gauss(0.08, 0.03)) # concave points
            row.append(random.gauss(0.20, 0.03)) # symmetry
            row.append(random.gauss(0.07, 0.01)) # fractal dimension
            
        # Generate Pathology Notes
        if diagnosis == 0:
            notes = [
                "The tumor appears small with a regular, round shape.",
                "Margins are smooth and well-defined. No signs of invasion.",
                "Benign appearance with uniform texture and low density.",
                "Likely a fibroadenoma. No suspicious features observed."
            ]
            row.append(random.choice(notes))
        else:
            notes = [
                "Large, irregular mass with spiculated margins.",
                "High suspicion of malignancy due to rapid growth and density.",
                "Invasive ductal carcinoma characteristics observed.",
                "Texture is heterogeneous with microcalcifications present."
            ]
            row.append(random.choice(notes))
            
        row.append(diagnosis)
        data.append(row)
        
    return feature_names + ['pathology_notes'], data

if __name__ == "__main__":
    print("Generating synthetic cancer data with pure Python...")
    try:
        header, rows = generate_cancer_data()
        output_path = "../data/cancer_data.csv"
        
        os.makedirs("../data", exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error generating data: {e}")
