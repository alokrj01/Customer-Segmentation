# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# 1. Initialize FastAPI App
app = FastAPI(title="Customer Segmentation API", version="1.0")

# 2. Load the Saved Model & Scaler
try:
    with open("customer_segmentation_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['features'] # To ensure correct column order
    
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_segmentation_model.pkl' not found.")
    raise Exception("Model file not found. Please upload the .pkl file.")

# 3. Define Input Data Schema
# We use a dictionary for brands to avoid writing 35 fields manually
class CustomerInput(BaseModel):
    Gender: str  # "M" or "F"
    Orders: int
    Brands: dict[str, int] = {} # Example: {"Jordan": 1, "Samsung": 0}

# 4. Root Endpoint (Health Check)
@app.get("/")
def home():
    return {"message": "Customer Segmentation API is Running!"}

# 5. Prediction Endpoint
@app.post("/predict_segment")
def predict_segment(data: CustomerInput):
    try:
        # 1. Process Input
        gender_val = 1 if data.Gender.upper() == 'M' else 0
        input_dict = {feature: 0 for feature in feature_columns}
        input_dict['Gender'] = gender_val
        input_dict['Orders'] = data.Orders
        
        for brand, count in data.Brands.items():
            if brand in input_dict:
                input_dict[brand] = count
        
        df_input = pd.DataFrame([input_dict])
        df_input = df_input[feature_columns] # Ensure correct order
        
        # 2. Scale & Predict
        scaled_data = scaler.transform(df_input)
        cluster_id = model.predict(scaled_data)[0]
        
        # Get Cluster Profile (Real Data)
        # we defined 'Center' (Average) to that cluster for the model
        center_scaled = model.cluster_centers_[cluster_id]
        
        # convert Scaled to Original Numbers (Example: Orders: 5, Samsung: 2)
        center_original = scaler.inverse_transform([center_scaled])[0]
        
        # for Dictionary  {Brand: Average_Value}
        cluster_profile = dict(zip(feature_columns, center_original))
        # --- NEW CODE END ---

        # Optional: Segment Names
        cluster_names = {
            0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3"
        }
        
        return {
            "cluster_id": int(cluster_id),
            "segment_name": cluster_names.get(cluster_id, "Unknown"),
            "cluster_profile": cluster_profile, # this data goes to frontend
            "message": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run Server Logic (for debugging)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)