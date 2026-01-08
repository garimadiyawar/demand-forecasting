from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/forecast")
def forecast(features: dict):
    prediction = model.predict([list(features.values())])
    return {"forecast": float(prediction[0])}
