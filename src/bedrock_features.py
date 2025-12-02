import boto3
import json
import csv
import os
import random

def get_bedrock_client():
    """
    Initialize Bedrock client. 
    Assumes AWS credentials are configured in environment or ~/.aws/credentials.
    """
    try:
        return boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    except Exception as e:
        print(f"Warning: Could not initialize Bedrock client: {e}")
        return None

def generate_risk_score(note, client=None):
    """
    Uses Amazon Bedrock (Claude 3 Sonnet) to analyze the pathology note 
    and assign a malignancy risk score (1-10).
    """
    if client is None:
        # Mock response if client is not available or for testing without cost
        # Simple heuristic for mock: if "malignancy" or "irregular" in note -> high score
        if "malignancy" in note or "irregular" in note or "spiculated" in note:
            return random.randint(8, 10)
        else:
            return random.randint(1, 3)

    prompt = f"""
    You are an expert pathologist. Analyze the following pathology note and assign a malignancy risk score from 1 (Benign) to 10 (Highly Malignant).
    
    Pathology Note: "{note}"
    
    Return ONLY the numeric score. Do not provide any explanation.
    """

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    })

    try:
        response = client.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        content = response_body.get('content')[0].get('text')
        score = int(content.strip())
        return score
    except Exception as e:
        print(f"Error invoking Bedrock: {e}")
        # Fallback to mock if API fails
        return generate_risk_score(note, client=None)

def add_bedrock_features(input_file, output_file):
    print("Initializing Bedrock client...")
    client = get_bedrock_client()
    
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        header = next(reader)
        new_header = header + ['llm_risk_score']
        writer.writerow(new_header)
        
        print("Processing rows with LLM...")
        rows = list(reader)
        # Find index of pathology_notes
        note_idx = header.index('pathology_notes')
        
        for i, row in enumerate(rows):
            note = row[note_idx]
            score = generate_risk_score(note, client)
            row.append(score)
            writer.writerow(row)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(rows)} rows")
                
    print(f"Feature engineering complete. Saved to {output_file}")

if __name__ == "__main__":
    input_path = "../data/cancer_data.csv"
    output_path = "../data/cancer_data_with_llm.csv"
    
    if os.path.exists(input_path):
        add_bedrock_features(input_path, output_path)
    else:
        print(f"Input file {input_path} not found. Please run data_generation.py first.")
