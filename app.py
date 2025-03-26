from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle  # Use pickle instead of joblib
import pandas as pd
import numpy as np

# Load the model, scaler, and label encoders using pickle
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Define the input data model using Pydantic
class DiabetesPredictionInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(data: DiabetesPredictionInput):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([dict(data)])

        # Encode categorical variables using the saved label encoders
        for column, le in label_encoders.items():
            input_data[column] = le.transform(input_data[column])

        # Standardize the features using the saved scaler
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)