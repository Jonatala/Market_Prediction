from fastapi import FastAPI
import  joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = "../../models/random_forest_model.pkl"

# Load model

model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Market Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    """Expect JSON input with feature values"""

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    #return {"prediction": float(prediction)}
    return {"prediction": round(float(prediction), 2)}
