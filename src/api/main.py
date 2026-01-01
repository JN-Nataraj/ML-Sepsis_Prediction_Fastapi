from typing import Union
from fastapi import FastAPI
from src.custom_threshold_classifier import ThresholdClassifier
import joblib
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

app = FastAPI()

model = None

class Model_Info(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    model_features: list[str]
    model_author: str
    model_description: str
    model_accuracy: float
    model_precision: float
    model_recall: float
    model_f1_score: float

class Predict_request(BaseModel):
    patient_id: str
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    M11: float
    BD2: float
    Age: int
    Insurance: int

class Predict_response(BaseModel):
    patient_id: str
    sepsis_prediction_class: str
    sepsis_prediction: int
    probability: float
    timestamp: str


def load_model():
    global model
    model = joblib.load("Models/Model_Balanced_45_Threshold.pkl")
    return model


@app.on_event("startup")
def startup_event():
    load_model()
    print("model Loaded Successfully")

@app.get("/")
def read_root():
    return{
        "message" : "Welcome to the Sepsis Prediction API. Please use the /predict endpoint for predicting the patient",
        "version" : "1.0.0",
        "author" : "J N Nataraj",
        "status" : "Development Environment",
        "endpoints" : {
            "docs": "/docs",
            "modelinfo":"/modelinfo",
            "predict" : "/predict"
        }
    }

@app.get("/modelinfo")
def get_model_info() -> Model_Info:
    if model is None:
        return "Model is not loaded properly"
    else:
        return Model_Info(
                model_name= "Sepsis Prediction Model",
                model_type= "Logistic Regression With Threshold Classifier 0.45",
                model_version= "1.0.0",
                model_features= ['ID', 'PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance','Sepssis'],
                model_author= "J N Nataraj",
                model_description= "A machine learning model to predict sepsis in patients based on various health parameters.",
                model_accuracy= 0.71,
                model_precision= 0.57,
                model_recall= 0.79,
                model_f1_score= 0.66
            )

@app.post("/predict", response_model=Predict_response)
def predict_sepsis(request: Predict_request) -> Predict_response:
    if model is None:
        return "Model is not loaded properly"
    else:
        data = request.dict()
        patient_id = data.pop("patient_id")
        data_df = pd.DataFrame([data])
        prediction = model.predict(data_df)
        prediction_proba = model.predict_proba(data_df)[0, 1]
        sepsis_class = "Positive" if prediction[0] == 1 else "Negative"
        timestamp = datetime.now().isoformat()
        return Predict_response(
            patient_id=patient_id,
            sepsis_prediction_class=sepsis_class,
            sepsis_prediction=int(prediction[0]),
            probability=float(prediction_proba),
            timestamp=timestamp
        )