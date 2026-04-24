from fastapi import FastAPI
import  joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = "../../models/randon_forest_model.pkl"

@app.get("/")
def home():
    return {"message": "Market Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    """Expect JSON input with feature values"""

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return {"prediction": float(prediction)}