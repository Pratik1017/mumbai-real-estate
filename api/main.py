
from fastapi import FastAPI
from api.schemas import InputCheck
from fastapi.responses import JSONResponse
import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Mumbai Real Estate Prediction API")


@app.get("/")
def home():
    return {"message": "Mumbai Real Estate API is running"}


@app.post("/predict")
def predict(data: InputCheck):
    user_input = pd.DataFrame([{
        "area": data.area,
        "bedrooms": data.bedrooms,
        "bathrooms": data.bathrooms,
        "status": data.status,
        "neworold": data.neworold,
        "furnished_status": data.furnished_status,
        "type_of_building": data.type_of_building,
        "category": data.category
    }])

    prediction = model.predict(user_input)[0]

    return JSONResponse(
        status_code=200,
        content={"predicted_price": float(prediction)}
    )
