from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

class HouseInput(BaseModel):
    location: str
    total_sqft: float
    bath: float
    balcony: float

@app.get("/")
def home():
    return {"message": "Bengaluru House Price Prediction API"}

@app.post("/predict")
def predict(data: HouseInput):
    input_data = pd.DataFrame([{
        "location": data.location,
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony
    }])

    prediction = model.predict(input_data)

    return {
        "predicted_price_lakhs": round(float(prediction[0]), 2)
    }