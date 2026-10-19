"""Fonctions modulaires pour le projet Churn (prédiction du départ client)."""

import glob
import json
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

TARGET = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# Configuration MLflow : base SQLite locale + nom de l'expérience
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn_prediction"
REGISTERED_MODEL_NAME = "churn_random_forest"


def configure_mlflow(tracking_uri=TRACKING_URI, experiment_name=EXPERIMENT_NAME):
    """Configure MLflow (backend de stockage et expérience)."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def find_data_file(pattern="Churn_Modelling*"):
    """Trouve le fichier de données (csv ou excel) dans le dossier courant."""
    files = [
        f for f in glob.glob(pattern) if f.lower().endswith((".csv", ".xlsx", ".xls"))
    ]
    if not files:
        raise FileNotFoundError("Fichier de données introuvable dans le dossier.")
    return files[0]


def load_data(path=None):
    path = path or find_data_file()
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def prepare_data(path=None, test_size=0.2, seed=42):
    """Retourne X_train, X_test, y_train, y_test."""
    df = load_data(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def log_model_to_mlflow(model, X_train):
    """Enregistre le modèle dans MLflow (et dans le registre de modèles)."""
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    try:
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
    except TypeError:  # anciennes versions de MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )


def train_model(X_train, y_train, seed=42, n_estimators=100, max_depth=None):
    """Entraîne un Random Forest et suit l'expérience avec MLflow."""
    configure_mlflow()
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": seed,
    }

    with mlflow.start_run(run_name="training"):
        for name, value in params.items():
            mlflow.log_param(name, value)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        mlflow.log_metric("train_accuracy", float(model.score(X_train, y_train)))
        log_model_to_mlflow(model, X_train)
    return model


def save_model(model, path="model.joblib"):
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="model.joblib"):
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    """Calcule les métriques, les affiche et les enregistre dans MLflow."""
    configure_mlflow()
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    with mlflow.start_run(run_name="evaluation"):
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        # Artefact : fichier JSON des métriques
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "metrics.json"
            artifact.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact))
    return metrics


def predict(model, X):
    return model.predict(X)
