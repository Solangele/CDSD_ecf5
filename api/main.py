from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import mlflow.sklearn
from contextlib import asynccontextmanager

# Schéma pour un client unique
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)

# Global pour stocker le modèle
MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    mlflow.set_tracking_uri("http://mlflow:5000")
    try:
        model_name = "churnguard"
        stage = "Production"
        model_uri = f"models:/{model_name}/{stage}"
        
        MODELS["model"] = mlflow.sklearn.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(model_name, stages=[stage])[0]
        MODELS["version"] = latest.version
        print(f"Modèle {model_name} v{MODELS['version']} chargé.")
    except Exception as e:
        print(f"ERREUR CRITIQUE chargement modèle : {e}")
        MODELS["model"] = None
        MODELS["version"] = "unknown"

    yield
    # Nettoyage si besoin
    MODELS.clear()
    print("Nettoyage effectué.")

app = FastAPI(title="ChurnGuard API", lifespan=lifespan)



@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    """
    Intercepte toutes les ValueError levées dans le code métier
    et les transforme en erreur HTTP 400.
    """
    message = str(exc)
    
    # Si l'erreur concerne la disponibilité du modèle -> 503
    if "disponible" in message or "chargé" in message:
        status_code = 503
    else:
        # Sinon, c'est une erreur client (batch vide, etc.) -> 400
        status_code = 400
        
    return JSONResponse(
        status_code=status_code,
        content={"message": message},
    )


@app.get("/health")
async def health():
    if not MODELS.get("model"):
        raise ValueError("Le service est indisponible : modèle non chargé.")
    return {
        "status": "ok", 
        "model": "churnguard", 
        "version": MODELS.get("version")
    }

@app.post("/predict")
async def predict(customer: CustomerData):
    if not MODELS.get("model"):
        raise ValueError("Modèle non disponible.")

    df = pd.DataFrame([customer.model_dump()])
    res_pred = MODELS["model"].predict(df)
    prediction = int(res_pred[0])

    try:
        probs = MODELS["model"].predict_proba(df)
        prob_churn = float(probs[0][1])
    except:
        prob_churn = 1.0 if prediction == 1 else 0.0

    return {
        "churn": bool(prediction),
        "probability": round(prob_churn, 4)
    }



@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    if not 1 <= len(customers) <= 100:
        raise ValueError("La taille du lot (batch) doit être entre 1 et 100 clients.")
    
    if not MODELS.get("model"):
        raise ValueError("Modèle non disponible")

    df = pd.DataFrame([c.model_dump() for c in customers])
    
    probs_churn = MODELS["model"].predict(df)[:, 1]
    preds = MODELS["model"].predict(df)
    
    return [
        {"churn": bool(p), "probability": round(float(pr), 4)}
        for p, pr in zip(preds, probs_churn)
    ]