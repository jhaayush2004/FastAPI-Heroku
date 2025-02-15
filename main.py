from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import numpy as np
from pathlib import Path

app = FastAPI()

# CORS Middleware (Allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema
class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the model with a dynamic path
model_path = Path(__file__).parent / "diabetes_model.sav"
diabetes_model = pickle.load(open(model_path, "rb"))

@app.post("/diabetes_prediction")
def diabetes_pred(input_parameters: ModelInput):
    input_dict = input_parameters.dict()
    
    # Convert input data into a NumPy array
    input_list = np.array([list(input_dict.values())]).reshape(1, -1)

    # Make prediction
    prediction = diabetes_model.predict(input_list)

    # Return JSON response
    return {"prediction": "Diabetic" if prediction[0] == 1 else "Not Diabetic"}
