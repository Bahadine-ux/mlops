"""Atelier 4 : exposition de la fonction predict() via FastAPI."""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import load_model, predict

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(title="API Churn", version="1.0")

# Chargement du modèle entraîné depuis le disque, au démarrage de l'API
try:
    model = load_model(str(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH.name}")
except Exception as exc:  # modèle absent ou illisible
    model = None
    print(f"Impossible de charger le modèle : {exc}")


class CustomerFeatures(BaseModel):
    """Caractéristiques d'un client."""

    CreditScore: int = Field(..., examples=[619])
    Geography: str = Field(..., examples=["France"])
    Gender: str = Field(..., examples=["Female"])
    Age: int = Field(..., examples=[42])
    Tenure: int = Field(..., examples=[2])
    Balance: float = Field(..., examples=[0.0])
    NumOfProducts: int = Field(..., examples=[1])
    HasCrCard: int = Field(..., examples=[1])
    IsActiveMember: int = Field(..., examples=[1])
    EstimatedSalary: float = Field(..., examples=[101348.88])


def to_model_input(features: CustomerFeatures) -> pd.DataFrame:
    """Applique le même encodage qu'à l'entraînement."""
    df = pd.DataFrame([features.model_dump()])
    df = pd.get_dummies(df)
    # Aligne les colonnes sur celles vues à l'entraînement
    df = df.reindex(columns=list(model.feature_names_in_), fill_value=0)
    return df.astype(float)


@app.get("/")
def home():
    return {"message": "API Churn en ligne. Documentation : /docs"}


@app.post("/predict")
def predict_route(features: CustomerFeatures):
    """Prédit si le client va partir (1) ou rester (0)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé : vérifiez model.joblib")
    try:
        X = to_model_input(features)
        prediction = int(predict(model, X)[0])
        probability = float(model.predict_proba(X)[0][1])
        return {"prediction": prediction, "probability": round(probability, 4)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {exc}")