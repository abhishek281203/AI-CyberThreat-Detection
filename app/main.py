from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
import pandas as pd
import os
import sys
import uvicorn
import json

# Add the app directory to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.preprocessing import preprocess_input_data

# Initialize FastAPI app
app = FastAPI(
    title="Cybersecurity Threat Detection API",
    description="ML-based API for detecting cyber threats",
    version="1.0.0"
)

# Load the model on startup
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "model.pkl")
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    model = None
    print(f"Error loading model: {str(e)}")

# Define input schema for prediction
class PredictionInput(BaseModel):
    """
    Schema for input data. The field names match the dataset features.
    All fields are defined as Optional to handle missing values in preprocessing.
    """
    class Config:
        extra = "allow"  # Allow extra fields that might not be in the schema

    # Example fields using the Field function for field names with spaces
    destination_port: Optional[int] = Field(None, alias="Destination Port")
    flow_duration: Optional[int] = Field(None, alias="Flow Duration")
    total_length_of_fwd_packets: Optional[int] = Field(None, alias="Total Length of Fwd Packets")
    fwd_packet_length_max: Optional[int] = Field(None, alias="Fwd Packet Length Max")
    # Additional fields will be handled through the Config.extra = "allow"

class PredictionOutput(BaseModel):
    prediction: str
    prediction_probability: float
    prediction_details: Dict[str, Any]

@app.get("/")
def read_root():
    """Root endpoint to check if API is running"""
    return {"message": "Cybersecurity Threat Detection API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint to verify API is online"""
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Endpoint to predict if a single network flow is an attack or normal
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert input to DataFrame
        input_dict = dict(input_data)
        
        # Convert aliased field names back to their original form
        for field_name, field_info in PredictionInput.__fields__.items():
            if field_info.alias and field_info.alias in input_dict:
                input_dict[field_info.alias] = input_dict.pop(field_name)
        
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess input data
        processed_input = preprocess_input_data(input_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_input)[0]
        prediction_class = model.predict(processed_input)[0]
        
        # Convert prediction to human-readable format
        result = "Attack" if prediction_class == 1 else "Normal"
        attack_probability = float(prediction_proba[1])
        
        return {
            "prediction": result,
            "prediction_probability": attack_probability,
            "prediction_details": {
                "raw_probabilities": [float(p) for p in prediction_proba],
                "class_label": int(prediction_class)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=List[PredictionOutput])
def batch_predict(input_data_list: List[PredictionInput]):
    """
    Endpoint to predict if multiple network flows are attacks or normal
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Process all inputs
        results = []
        for input_data in input_data_list:
            # Convert input to DataFrame
            input_dict = dict(input_data)
            
            # Convert aliased field names back to their original form
            for field_name, field_info in PredictionInput.__fields__.items():
                if field_info.alias and field_info.alias in input_dict:
                    input_dict[field_info.alias] = input_dict.pop(field_name)
            
            input_df = pd.DataFrame([input_dict])
            
            # Preprocess input data
            processed_input = preprocess_input_data(input_df)
            
            # Make prediction
            prediction_proba = model.predict_proba(processed_input)[0]
            prediction_class = model.predict(processed_input)[0]
            
            # Convert prediction to human-readable format
            result = "Attack" if prediction_class == 1 else "Normal"
            attack_probability = float(prediction_proba[1])
            
            results.append({
                "prediction": result,
                "prediction_probability": attack_probability,
                "prediction_details": {
                    "raw_probabilities": [float(p) for p in prediction_proba],
                    "class_label": int(prediction_class)
                }
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 