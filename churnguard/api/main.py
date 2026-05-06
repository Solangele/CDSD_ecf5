from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import mlflow.pyfunc
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
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Chargement au démarrage
    try:
        model_name = "churnguard"
        stage = "Production"
        model_uri = f"models:/{model_name}/{stage}"
        MODELS["model"] = mlflow.sklearn.load_model(model_uri)
        # On récupère la version pour le /health
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(model_name, stages=[stage])[0]
        MODELS["version"] = latest.version
        print(f"Modèle {model_name} v{MODELS['version']} chargé.")
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        MODELS["model"] = None
    yield
    # Nettoyage si besoin
    MODELS.clear()

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
    
    # Transformation en DataFrame (format attendu par scikit-learn)
    df = pd.DataFrame([customer.model_dump()])
    
    # Prédiction
    prob = MODELS["model"].predict_proba(df)
    prob_churn = prob[0][1]
    prediction = MODELS["model"].predict(df)[0]
    
    return {
        "churn": bool(prediction),
        "probability": round(float(prob_churn), 4)
    }

@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    if not 1 <= len(customers) <= 100:
        raise ValueError("La taille du lot (batch) doit être entre 1 et 100 clients.")
    
    if not MODELS.get("model"):
        raise ValueError("Modèle non disponible")

    df = pd.DataFrame([c.model_dump() for c in customers])
    
    probs_churn = MODELS["model"].predict_proba(df)[:, 1]
    preds = MODELS["model"].predict(df)
    
    return [
        {"churn": bool(p), "probability": round(float(pr), 4)}
        for p, pr in zip(preds, probs_churn)
    ]